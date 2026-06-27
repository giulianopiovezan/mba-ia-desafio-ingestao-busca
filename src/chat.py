from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from search import search_prompt

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.1,
  )


def main():
    while True:
        try:
     
            print("\nFaça sua pergunta:")
            pergunta = input("\nPERGUNTA: ").strip()
            
    
            print("\nBuscando informações...")
                 
            try:
                prompt = search_prompt(pergunta)

                pipeline = prompt | llm | StrOutputParser()

                response = pipeline.invoke({})
                print(f"RESPOSTA: {response}")
            except Exception as e:
                print(f"Erro ao processar pergunta: {e}")
                print("Tente novamente com uma pergunta diferente.")
            
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n\nChat interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"\nErro inesperado: {e}")       

if __name__ == "__main__":
    main()