import os
import asyncio
import platform

import anthropic
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import workflow, tool, task, agent, llm

anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
datadog_api_key = os.environ.get("DD_API_KEY")
use_llm_agent = os.environ.get("DD_LLM_OBS_USE_AGENT")
if use_llm_agent == 'True' or use_llm_agent == 'true' or use_llm_agent == '1':
    use_llm_agent = True
else:
    use_llm_agent = False

if use_llm_agent:
    LLMObs.enable(ml_app='curryware_ml', api_key=datadog_api_key, site="datadoghq.com",
                    agentless_enabled=False, env='prod',
                    service='curryware_llm_service')
else:
    LLMObs.enable(ml_app='curryware_ml', api_key=datadog_api_key, site="datadoghq.com",
                  agentless_enabled=True, env='prod',
                  service='curryware_llm_service')


def main():
    print('Anthropic API Key: -', anthropic_api_key)
    build_prompt_input()


@workflow
def build_prompt_input():
    report_input = get_letter_text()
    system_prompt = get_system_prompt(report_input)
    anthropic_response =asyncio.run(make_llm_call(system_prompt, report_input))
    output = anthropic_response.content[0].text
    print(output)


@tool
def get_letter_text():
    filename = 'gm_chairman_letter.txt'
    file_content = open(filename).read()

    return file_content


@task
def get_system_prompt(file_content):
    system_prompt = 'You are an expert research assistant. Here is a document you will '
    system_prompt = system_prompt + 'answer questions about: '
    system_prompt = system_prompt + ' '.format(file_content)
    system_prompt = system_prompt + '\nFirst, find the quotes from the document that are '
    system_prompt = system_prompt + 'most relevant to answering the question, and then '
    system_prompt = system_prompt + 'print them in numbered order. Quotes should be '
    system_prompt = system_prompt + 'relatively short.\n'
    system_prompt = system_prompt + 'If there are no relevant quotes, write “No relevant '
    system_prompt = system_prompt + 'quotes” instead.\n'
    system_prompt = system_prompt + 'Then, answer the question, starting with “Answer:“. '
    system_prompt = system_prompt + 'Do not include or reference quoted content'
    system_prompt = system_prompt + 'verbatim in the answer. Don’t say “According to '
    system_prompt = system_prompt + 'Quote [1]” when answering. Instead make references '
    system_prompt = system_prompt + 'to quotes relevant to each section of the answer '
    system_prompt = system_prompt + 'by adding their bracketed numbers at the end '
    system_prompt = system_prompt + 'of relevant sentences'

    return system_prompt

@llm(model_name='claude-3-haiku-20240307', name='curryware_llm',
     model_provider='anthropic')
# The decorator is not required because Anthropic is one of the Datadog LLM integrations.
# See https://docs.datadoghq.com/llm_observability/setup/sdk/#llm-span
async def make_llm_call(system_prompt, report_input) -> anthropic.types.message.Message:
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
    message = await client.messages.create(
        model='claude-3-haiku-20240307',
        max_tokens = 2048,
        system = system_prompt,
        messages=[
            {'role': 'user', 'content': report_input }
        ]
    )
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    host_name = platform.node()

    LLMObs.annotate(
        span = None,
        input_data = report_input,
        output_data = message.content[0].text,
        metadata = {'temperature': 0, 'max_tokens': 2048},
        metrics = {'input_tokens': input_tokens, 'output_tokens': output_tokens,
                   'total_tokens': input_tokens + output_tokens},
        tags = {'host': host_name}
    )
    span_context = LLMObs.export_span(span=None)

    LLMObs.submit_evaluation(
        span_context = span_context,
        label = 'evaluation',
        metric_type = 'score',
        value = 3.5
    )

    return message


if __name__ == "__main__":
    main()
