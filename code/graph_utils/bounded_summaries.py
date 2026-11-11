import asyncio
import time
from dataclasses import replace

from openai import LengthFinishReasonError
from pydantic import ValidationError
from ragu.common.logger import logger
from ragu.common.prompts.default_models import ClusterSummarizationModel
from ragu.common.prompts.messages import render
from ragu.graph.artifacts_summarizer import EntitySummarizer, RelationSummarizer
from ragu.graph.community_summarizer import CommunitySummarizer
from ragu.graph.types import Community, CommunitySummary, Entity, Relation


class SummaryBudgetError(RuntimeError):
    pass


class SummaryBudget:
    def __init__(self, llm):
        self.llm = llm
        self.limit = llm.context_tokens - llm.completion_tokens

    def size(self, conversation):
        return self.llm.count_prompt_tokens(conversation)

    def fits(self, conversation):
        # A token covers at least one byte of text, and the chat template adds a few special tokens per message, so
        # a prompt within the limit by its bytes is within it by its tokens: most of the millions of graph descriptions
        # need no tokenization.
        upper = sum(len(str(message.get('content', '')).encode()) + 64 for message in conversation)
        return upper <= self.limit or self.size(conversation) <= self.limit

    @staticmethod
    def split(text):
        middle = len(text) // 2
        boundary = text.rfind('\n', len(text) // 4, middle + 1)
        if boundary < 0:
            boundary = text.rfind(' ', len(text) // 4, middle + 1)
        middle = min(len(text) - 1, max(1, boundary + 1 if boundary >= 0 else middle))
        return text[:middle], text[middle:]

    async def recover(self, text, renderer, schema, field, *, label, depth=0):
        conversation = renderer(text)
        error = 'Уточнение промпта не помещается в контекст.'
        for attempt in range(1, 3):
            prompt = [*conversation, {'role': 'user', 'content': (
                f'Верни краткий ответ по заданной JSON-схеме. Каждое текстовое поле — не более '
                f'{1000 // attempt} символов, каждый список — не более 5 пунктов. '
                'Выдели основные факты без повторов, формул и рассуждений.'
            )}]
            if self.size(prompt) > self.limit:
                break
            logger.warning('[ГРАФ: ПОВТОР ОБОБЩЕНИЯ] {}: уровень {}, попытка {}/2; запрашиваю краткий ответ.',
                           label, depth, attempt)
            try:
                result = await self.llm.chat_completion(prompt, output_schema=schema, temperature=0.2)
                if isinstance(getattr(result, field, None), str) and getattr(result, field).strip():
                    return result
                error = 'Модель вернула пустое описание.'
            except (LengthFinishReasonError, ValidationError) as exc:
                error = type(exc).__name__
        if depth >= 4 or len(text) <= 1:
            raise SummaryBudgetError(f'Не удалось обобщить {label} после ограниченных повторов: {error}')
        logger.warning('[ГРАФ: ПОВТОР ОБОБЩЕНИЯ] {}: {} Разбиваю материал на две меньшие части; уровень {}.',
                       label, error, depth + 1)
        results = [await self.recover(part, renderer, schema, field, label=label, depth=depth + 1)
                   for part in self.split(text)]
        material = '\n'.join(getattr(result, field) for result in results)
        if self.size(renderer(material)) >= self.size(conversation):
            raise SummaryBudgetError(f'Повторное обобщение {label} не сокращает материал; результат не сохранён.')
        return await self.recover(material, renderer, schema, field, label=label, depth=depth + 1)

    async def batch(self, conversations, schema, field, *, recover, desc=None):
        responses = await self.llm.batch_chat_completion(
            conversations, output_schema=schema, continue_on_error=True, desc=desc,
        )
        if len(responses) != len(conversations):
            raise SummaryBudgetError('Потеряна часть ответов при обобщении графа.')
        failed = [index for index, response in enumerate(responses)
                  if not isinstance(getattr(response, field, None), str) or not getattr(response, field).strip()]

        async def retry(index):
            responses[index] = await recover(index)

        await self.parallel(failed, retry)
        return responses

    async def reduce(self, text, renderer, schema, field, *, label, final_renderer=None):
        final_renderer = final_renderer or renderer
        initial = self.size(final_renderer(text))
        if initial <= self.limit:
            return text
        logger.warning('[ГРАФ: ОБОБЩЕНИЕ] {}: {} входных токенов; допустимо {}. Разбиваю на части.',
                       label, initial, self.limit)
        overhead = max(self.size(renderer('')), self.size(final_renderer('')))
        if overhead >= self.limit:
            raise SummaryBudgetError(f'Служебная часть промпта {label} не помещается в контекст.')
        target = overhead + (self.limit - overhead) // 2
        current = text
        previous = initial
        for level in range(1, 9):
            pending = [current]
            parts = []
            while pending:
                part = pending.pop()
                if self.size(renderer(part)) <= target:
                    parts.append(part)
                    continue
                if len(part) <= 1:
                    raise SummaryBudgetError(f'Невозможно разбить материал {label} в пределах контекста.')
                pending.extend(reversed(self.split(part)))
            logger.info('[ГРАФ: ОБОБЩЕНИЕ] {}: уровень {}, частей {}.', label, level, len(parts))

            async def recover(index, parts=parts):
                return await self.recover(parts[index], renderer, schema, field, label=label)

            responses = await self.batch(
                [renderer(part) for part in parts], schema, field, recover=recover,
                desc=f'Обобщение больших описаний: {label}',
            )
            if len(responses) != len(parts):
                raise SummaryBudgetError(f'Потеряна часть ответа при обобщении {label}.')
            summaries = [getattr(response, field, None) for response in responses]
            if any(not isinstance(value, str) or not value.strip() for value in summaries):
                raise SummaryBudgetError(f'Пустой ответ при обобщении {label}; результат не сохранён.')
            current = '\n'.join(summaries)
            size = self.size(final_renderer(current))
            if size <= self.limit:
                logger.info('[ГРАФ: ОБОБЩЕНИЕ] {}: {} → {} входных токенов.', label, initial, size)
                return current
            if size >= previous:
                raise SummaryBudgetError(f'Размер обобщения {label} не уменьшается; результат не сохранён.')
            previous = size
        raise SummaryBudgetError(f'Превышено число уровней обобщения {label}; результат не сохранён.')

    async def parallel(self, items, operation):
        pending = iter(items)

        async def worker():
            for item in pending:
                await operation(item)

        workers = [asyncio.create_task(worker()) for _ in range(min(8, len(items)))]
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def descriptions(self, items, instruction, argument, language):
        def conversation(item):
            return render(instruction.messages, **{argument: [item]}, language=language)[0].to_openai()

        oversized = [item for item in items if not self.fits(conversation(item))]

        async def reduce(item):
            def renderer(text):
                return conversation(replace(item, description=text))

            item.description = await self.reduce(
                item.description, renderer, instruction.pydantic_model, 'description',
                label=item.entity_name if isinstance(item, Entity) else f'{item.subject_name} → {item.object_name}',
            )

        await self.parallel(oversized, reduce)


class BudgetedEntitySummarizer(EntitySummarizer):
    async def merged_descriptions(self, grouped_entities_df):
        """The description of every group, as the library merges it (joined, or clustered and summarized). Collected
        in a list: a DataFrame updated row by row copies its column each time (about 80 groups a second on 631K)."""
        started = reported = time.monotonic()
        total = len(grouped_entities_df)
        logger.info('[ГРАФ: ОПИСАНИЯ СУЩНОСТЕЙ] Подготовка {} групп описаний.', total)
        descriptions = []
        for completed, value in enumerate(grouped_entities_df['description'].tolist(), 1):
            descriptions.append(await self._summarize_by_cluster_if_needed(value))
            now = time.monotonic()
            if now - reported >= 30 or completed == total:
                logger.info('[ГРАФ: ОПИСАНИЯ СУЩНОСТЕЙ] {}/{} групп; прошло {:.0f} с.',
                            completed, total, now - started)
                reported = now
        return descriptions

    async def summarize_entities(self, grouped_entities_df):
        if not self.use_llm_summarization:
            return await super().summarize_entities(grouped_entities_df)
        grouped_entities_df['description'] = await self.merged_descriptions(grouped_entities_df)
        mask = grouped_entities_df['duplicate_count'].to_numpy() > self.summarize_only_if_more_than
        singles = [Entity(**row) for _, row in grouped_entities_df.loc[~mask].drop('duplicate_count', axis=1).iterrows()]
        multiples = [Entity(**row) for _, row in grouped_entities_df.loc[mask].drop('duplicate_count', axis=1).iterrows()]
        instruction = self.get_prompt('entity_summarizer')
        budget = SummaryBudget(self.llm)
        await budget.descriptions(singles + multiples, instruction, 'entity', self.language)
        if multiples:
            conversations = render(instruction.messages, entity=multiples, language=self.language)

            async def recover(index):
                entity = multiples[index]

                def renderer(text):
                    return render(instruction.messages, entity=[replace(entity, description=text)],
                                  language=self.language)[0].to_openai()

                return await budget.recover(entity.description, renderer, instruction.pydantic_model,
                                            'description', label=entity.entity_name)

            responses = await budget.batch(
                [conversation.to_openai() for conversation in conversations],
                instruction.pydantic_model, 'description', recover=recover, desc='Entity summarization',
            )
            for entity, response in zip(multiples, responses, strict=True):
                if response is not None and getattr(response, 'description', None) is not None:
                    entity.description = response.description
        return singles + multiples


class BudgetedRelationSummarizer(RelationSummarizer):
    async def summarize_relations(self, grouped_relations_df):
        if not self.use_llm_summarization:
            return await super().summarize_relations(grouped_relations_df)
        mask = grouped_relations_df['duplicate_count'].to_numpy() > self.summarize_only_if_more_than
        singles = [Relation(**row) for _, row in grouped_relations_df.loc[~mask].drop('duplicate_count', axis=1).iterrows()]
        multiples = [Relation(**row) for _, row in grouped_relations_df.loc[mask].drop('duplicate_count', axis=1).iterrows()]
        instruction = self.get_prompt('relation_summarizer')
        budget = SummaryBudget(self.llm)
        await budget.descriptions(singles + multiples, instruction, 'relation', self.language)
        if multiples:
            conversations = render(instruction.messages, relation=multiples, language=self.language)

            async def recover(index):
                relation = multiples[index]

                def renderer(text):
                    return render(instruction.messages, relation=[replace(relation, description=text)],
                                  language=self.language)[0].to_openai()

                return await budget.recover(relation.description, renderer, instruction.pydantic_model,
                                            'description', label=f'{relation.subject_name} → {relation.object_name}')

            responses = await budget.batch(
                [conversation.to_openai() for conversation in conversations],
                instruction.pydantic_model, 'description', recover=recover, desc='Relation summarization',
            )
            for relation, response in zip(multiples, responses, strict=True):
                if response is not None and getattr(response, 'description', None) is not None:
                    relation.description = response.description
        return singles + multiples


class BudgetedCommunitySummarizer(CommunitySummarizer):
    async def summarize(self, communities):
        ordered = [Community(
            level=community.level, cluster_id=community.cluster_id, id=community.id,
            entities=sorted(community.entities, key=lambda entity: entity.id),
            relations=sorted(community.relations, key=lambda relation: relation.id),
        ) for community in communities]
        instruction = self.get_prompt('community_report')

        def conversation(community):
            return render(instruction.messages, community=[community], language=self.language)[0].to_openai()

        conversations = [conversation(community) for community in ordered]
        materials = {}
        budget = SummaryBudget(self.llm)
        oversized = [index for index, value in enumerate(conversations) if budget.size(value) > budget.limit]

        def material(community):
            return '\n'.join(
                [f'Entity: {entity.entity_name}, description: {entity.description}' for entity in community.entities]
                + [f'{relation.subject_name} -> {relation.object_name}, relations description: {relation.description}'
                   for relation in community.relations]
            )

        def report_renderer(community):
            def renderer(text):
                entity = Entity('Материалы сообщества', 'SUMMARY', text, [])
                return conversation(replace(community, entities=[entity], relations=[]))
            return renderer

        async def reduce(index):
            community = ordered[index]
            final_renderer = report_renderer(community)

            def renderer(material):
                return [{'role': 'user', 'content': (
                    'Обобщи материал части сообщества графа. Сохрани существенные факты, имена и связи; '
                    'не добавляй отсутствующие сведения. Пиши по-русски. Верни JSON с полем content.\n\n' + material
                )}]

            materials[index] = await budget.reduce(
                material(community), renderer, ClusterSummarizationModel, 'content',
                label=community.id, final_renderer=final_renderer,
            )
            conversations[index] = final_renderer(materials[index])

        await budget.parallel(oversized, reduce)

        async def recover_report(index):
            community = ordered[index]
            text = materials[index] if index in materials else material(community)
            return await budget.recover(text, report_renderer(community), instruction.pydantic_model,
                                        'summary', label=community.id)

        responses = await budget.batch(
            conversations, instruction.pydantic_model, 'summary', recover=recover_report, desc='Summarized communities',
        )
        output = []
        for community, response in zip(ordered, responses, strict=True):
            if response is None or getattr(response, 'title', None) is None:
                logger.warning('Пропущено сообщество {}: модель не вернула описание.', community.id)
                continue
            output.append(CommunitySummary(community.id, self.combine_report_text(response)))
        return output


def install_budgeted_summarizers(pipeline):
    if pipeline.llm is None or not hasattr(pipeline.llm, 'count_prompt_tokens'):
        return
    if pipeline.entity_summarizer is not None:
        old = pipeline.entity_summarizer
        pipeline.entity_summarizer = BudgetedEntitySummarizer(
            llm=old.llm, use_llm_summarization=old.use_llm_summarization, use_clustering=old.use_clustering,
            embedder=old.embedder, cluster_only_if_more_than=old.cluster_only_if_more_than,
            summarize_only_if_more_than=old.summarize_only_if_more_than, language=old.language,
        )
    if pipeline.relation_summarizer is not None:
        old = pipeline.relation_summarizer
        pipeline.relation_summarizer = BudgetedRelationSummarizer(
            llm=old.llm, use_llm_summarization=old.use_llm_summarization,
            summarize_only_if_more_than=old.summarize_only_if_more_than, language=old.language,
        )
    if pipeline.community_summarizer is not None:
        pipeline.community_summarizer = BudgetedCommunitySummarizer(pipeline.llm, language=pipeline.language)
