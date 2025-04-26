from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
  model="gpt-4.1",
  input=[
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "what is NGS?"
        }
      ]
    },
    {
      "id": "msg_6808f156c4108191a284e595e12402a00a112dc2038ef1ca",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "**NGS** stands for **Next-Generation Sequencing**.\n\n### What is Next-Generation Sequencing (NGS)?\nNGS refers to modern sequencing technologies that allow **rapid sequencing of DNA or RNA** much more efficiently and cost-effectively than traditional methods like Sanger sequencing.\n\n### Key Features:\n- **Massively Parallel:** Sequences millions of fragments simultaneously.\n- **High Throughput:** You can sequence entire genomes, exomes, transcriptomes, or targeted regions in a single experiment.\n- **Versatile Applications:** Used in genomics, transcriptomics, metagenomics, epigenomics, and more.\n\n### Main Applications:\n- **Whole genome sequencing (WGS)**\n- **Whole exome sequencing (WES)**\n- **RNA sequencing (RNA-seq)**\n- **Targeted gene sequencing**\n- **Detection of mutations, variants, and structural changes**\n- **Microbiome analysis**\n\n### Popular NGS Platforms:\n- **Illumina**\n- **Ion Torrent**\n- **PacBio**\n- **Oxford Nanopore**\n\n### Benefits:\n- **Speed:** Fast compared to older methods.\n- **Scale:** Large volumes of data in a single run.\n- **Cost:** Greatly reduced cost per base/gene.\n\n### Example Use Case:\nIn cancer research, NGS is used to **identify genetic mutations** in tumors to guide personalized therapy.\n\n---\n\n**In summary:**  \nNGS (Next-Generation Sequencing) is a powerful technology for analyzing genetic material, revolutionizing biological and medical research."
        }
      ]
    }
  ],
  text={
    "format": {
      "type": "text"
    }
  },
  reasoning={},
  tools=[],
  temperature=1,
  max_output_tokens=2048,
  top_p=1,
  store=True
)

print(response)