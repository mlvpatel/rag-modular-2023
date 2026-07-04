# Sample data

Three sample documents so anyone can run RagFlowPro and judge its performance without supplying their own files. Two are small hand written business documents, and one is a real public SEC filing, so you can see it work on both clean text and a dense real world document.

| File | Type | Use case it demonstrates |
|---|---|---|
| acme_employee_handbook.txt | HR policy | Internal knowledge and policy lookup |
| nimbus_product_faq.txt | Product FAQ | Customer support and pricing questions |
| nvidia_10k_excerpt.txt | Real SEC 10-K excerpt (public filing) | Finance and analyst questions over a real document |

## Load them

Start the stack first, then load the samples. For a fully local, no cost run with Ollama:

```bash
make db-up
ollama serve &
ollama pull nomic-embed-text
ollama pull llama3.2:3b
EMBEDDING_PROVIDER=ollama python scripts/load_sample_data.py
```

Or, once the stack is up, simply:

```bash
make load-samples
```

Then open the UI (`make frontend`, at http://localhost:8501), pick a model, and ask questions.

## Questions to try, with the expected answer

Use these to check performance. The answer should be grounded in the documents, and the last one should be declined rather than guessed.

Employee handbook:
- How many vacation days do new employees get? (15, rising to 25 after five years)
- What is the remote work policy? (up to three days per week, fully remote needs manager and HR approval)
- How many paid sick days are there? (10, separate from vacation)

Product FAQ:
- What does the Nimbus Pro plan cost? (49 dollars per month billed annually, or 59 month to month)
- What is the difference between the Free and Pro plans? (Free is one project and five thousand events, Pro is unlimited projects and two million events)
- What support do Enterprise customers get? (24 by 7 priority support with a 99.9 percent uptime guarantee)

NVIDIA 10-K:
- What are the building blocks of NVIDIA's platform? (GPUs, CPUs, CUDA, and networking)
- What are some of NVIDIA's risk factors? (it should summarize risks named in the filing)

Memory check (ask the second one right after the first, with no extra context):
- How many vacation days do new employees get?
- And what about after five years? (it should answer 25, using the earlier turn)

Honesty check (the answer is not in any document):
- What is Acme's parental leave policy? (it should say the documents do not cover this, not invent an answer)

## Note

These samples are for demonstration. For a real evaluation, load your own documents the same way and ask questions you already know the answers to.
