import asyncio
from dataclasses import replace

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
                middle = len(part) // 2
                boundary = part.rfind('\n', len(part) // 4, middle + 1)
                if boundary < 0:
                    boundary = part.rfind(' ', len(part) // 4, middle + 1)
                middle = min(len(part) - 1, max(1, boundary + 1 if boundary >= 0 else middle))
                pending.extend((part[middle:], part[:middle]))
            logger.info('[ГРАФ: ОБОБЩЕНИЕ] {}: уровень {}, частей {}.', label, level, len(parts))
            responses = await self.llm.batch_chat_completion(
                [renderer(part) for part in parts], output_schema=schema,
                continue_on_error=False, desc=f'Обобщение больших описаний: {label}',
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

        oversized = [item for item in items if self.size(conversation(item)) > self.limit]

        async def reduce(item):
            def renderer(text):
                return conversation(replace(item, description=text))

            item.description = await self.reduce(
                item.description, renderer, instruction.pydantic_model, 'description',
                label=item.entity_name if isinstance(item, Entity) else f'{item.subject_name} → {item.object_name}',
            )

        await self.parallel(oversized, reduce)


class BudgetedEntitySummarizer(EntitySummarizer):
    async def summarize_entities(self, grouped_entities_df):
        if not self.use_llm_summarization:
            return await super().summarize_entities(grouped_entities_df)
        for index, row in grouped_entities_df.iterrows():
            grouped_entities_df.loc[index, 'description'] = await self._summarize_by_cluster_if_needed(row['description'])
        mask = grouped_entities_df['duplicate_count'].to_numpy() > self.summarize_only_if_more_than
        singles = [Entity(**row) for _, row in grouped_entities_df.loc[~mask].drop('duplicate_count', axis=1).iterrows()]
        multiples = [Entity(**row) for _, row in grouped_entities_df.loc[mask].drop('duplicate_count', axis=1).iterrows()]
        instruction = self.get_prompt('entity_summarizer')
        await SummaryBudget(self.llm).descriptions(singles + multiples, instruction, 'entity', self.language)
        if multiples:
            conversations = render(instruction.messages, entity=multiples, language=self.language)
            responses = await self.llm.batch_chat_completion(
                [conversation.to_openai() for conversation in conversations],
                output_schema=instruction.pydantic_model, continue_on_error=True, desc='Entity summarization',
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
        await SummaryBudget(self.llm).descriptions(singles + multiples, instruction, 'relation', self.language)
        if multiples:
            conversations = render(instruction.messages, relation=multiples, language=self.language)
            responses = await self.llm.batch_chat_completion(
                [conversation.to_openai() for conversation in conversations],
                output_schema=instruction.pydantic_model, continue_on_error=True,
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
        budget = SummaryBudget(self.llm)
        oversized = [index for index, value in enumerate(conversations) if budget.size(value) > budget.limit]

        async def reduce(index):
            community = ordered[index]
            text = '\n'.join(
                [f'Entity: {entity.entity_name}, description: {entity.description}' for entity in community.entities]
                + [f'{relation.subject_name} -> {relation.object_name}, relations description: {relation.description}'
                   for relation in community.relations]
            )

            def final_renderer(material):
                entity = Entity('Материалы сообщества', 'SUMMARY', material, [])
                return conversation(replace(community, entities=[entity], relations=[]))

            def renderer(material):
                return [{'role': 'user', 'content': (
                    'Обобщи материал части сообщества графа. Сохрани существенные факты, имена и связи; '
                    'не добавляй отсутствующие сведения. Пиши по-русски. Верни JSON с полем content.\n\n' + material
                )}]

            material = await budget.reduce(
                text, renderer, ClusterSummarizationModel, 'content',
                label=community.id, final_renderer=final_renderer,
            )
            conversations[index] = final_renderer(material)

        await budget.parallel(oversized, reduce)
        responses = await self.llm.batch_chat_completion(
            conversations, output_schema=instruction.pydantic_model, continue_on_error=True,
            desc='Summarized communities',
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
