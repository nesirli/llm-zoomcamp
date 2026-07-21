import time

from starter import rag as base_rag
from rag_helper import RAGBase
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")


class RAGTraced(RAGBase):

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search") as span:
            span.set_attribute("query", query)
            span.set_attribute("num_results", num_results)
            start = time.perf_counter()
            results = super().search(query, num_results=num_results)
            duration_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("duration_ms", duration_ms)
            span.set_attribute("num_results_returned", len(results))
            return results

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            span.set_attribute("model", self.model)
            start = time.perf_counter()
            response = super().llm(prompt)
            duration_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("duration_ms", duration_ms)
            span.set_attribute("output_text", response.output_text)
            usage = response.usage
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)
            span.set_attribute("total_tokens", usage.total_tokens)
            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            span.set_attribute("query", query)
            start = time.perf_counter()

            search_results = self.search(query)
            prompt = self.build_prompt(query, search_results)
            response = self.llm(prompt)
            answer = response.output_text

            duration_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("duration_ms", duration_ms)
            span.set_attribute("answer", answer)

            usage = response.usage
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)
            span.set_attribute("total_tokens", usage.total_tokens)
            return answer


rag = RAGTraced(
    index=base_rag.index,
    llm_client=base_rag.llm_client,
    instructions=base_rag.instructions,
    prompt_template=base_rag.prompt_template,
    model=base_rag.model,
)

if __name__ == "__main__":
    query = "How does the agentic loop keep calling the model until it stops?"
    answer = rag.rag(query)
    print(answer)