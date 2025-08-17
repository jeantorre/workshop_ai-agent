import os

from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv

load_dotenv()

LLM = LLM(model="ollama/deepseek-r1:7b", base_url=os.getenv("OLLAMA_HOST"))


def agente_criador_conteudo_linkedin(topico: str) -> Crew:
    agente_linkedin = Agent(
        role="Criador de Conteúdo para LinkedIn",
        goal=(
            "Criar enganjamento para profissionais de tecnologia para o LinkedIn "
            f"sobre o tópico: {topico}."
        ),
        verbose=True,
        llm=LLM,
        backstory=(
            "Você é um escritor de conteúdo profissional especializado em "
            "criar enganjamento para profissionais de tecnologia para o LinkedIn. "
            "Escreva de forma cativante com um início forte para prender a atenção do "
            "leitor. Sempre escreva em português brasileiro."
        ),
        max_tokens=300,
    )
    task_criar_post = Task(
        description=(
            f"Criar um post para o LinkedIn sobre o tópico: {topico}. Foque em conteudo "
            "que seja interessante para os profissionais de tecnologia. Seja conciso e "
            "escreva algo entre 150 e 200 palavras. Ao final adicione uma chamada para "
            "ação do leitor."
        ),
        expected_output=(
            f"Um post pronto para ser publicado no LinkedIn sobre o tema {topico}."
        ),
        agent=agente_linkedin,
        verbose=True,
    )

    crew = Crew(
        agents=[agente_linkedin],
        tasks=[task_criar_post],
        process=Process(Process.sequential),
        verbose=True,
    )

    return crew


if __name__ == "__main__":
    topico = "Escreva um texto sobre agentes de IA"
    crew = agente_criador_conteudo_linkedin(topico)
    result = crew.kickoff()
    print(result)
