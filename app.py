import gradio as gr
from langchain.chat_models.gigachat import GigaChat
import os
from langchain_core.tools import tool
from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders
from langchain.agents import create_tool_calling_agent, AgentExecutor, create_gigachat_functions_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.graphs.networkx_graph import KG_TRIPLE_DELIMITER
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pyvis.network import Network
import networkx as nx
from langchain_community.document_loaders import UnstructuredFileLoader

api_key = os.getenv("SBER")

giga = GigaChat(credentials=api_key,
                model='GigaChat-Pro-preview',
                verify_ssl_certs=False,
                scope='GIGACHAT_API_CORP',
                profanity_check=False
                )

@tool
def get_papers(query: str, n: int) -> str:
  """Делает запрос для поиска n актуальных научных статей.\
  Запрос делай на английском, переводя запрос пользователя, если он задан по-русски.\
  Возвращает информацию о n статьях (названия, авторы, аннотации, темы, ключевые слова)."""

  try:
    papers = Works().search(query).filter(publication_year=2024).filter(is_oa=True).get()
    print(papers)
  except Exception as e:
    return f'Не удалось совершить запрос: {e}'
  txt = ''
  if len(papers) > 0:
    for i in range(len(papers)):
      txt += f'Запрос: {query}\n\n'
      txt += 'Название: ' + papers[i]['title'] + '\n\n'
      authors = papers[i]['authorships']
      txt += "Авторы:\n"
      for author in authors:
        txt += author['raw_author_name'] + '\n'
      abstract = papers[i]['abstract_inverted_index']
      abstract = format_abstract(abstract)
      txt += 'Аннотация:\n'
      txt += abstract + '\n'
      txt += 'Тематики:\n'
      for t in papers[i]['topics']:
        txt += t['display_name'] + '\n'
      txt += 'Ключевые слова:\n'
      for k in papers[i]['keywords']:
        txt += k['display_name'] + '\n'
  else:
    return f'Запрос не дал результатов: {papers}'

  return txt

@tool
def search(query: str) -> str:
    """Поиск информации в Интернете. Передает запрос, возвращает выдачу поисковика.\
    Полученный результат надо обобщить и передать пользователю."""
    result = search.run(query)
    return result

@tool
def switch_person() -> str:
  """Инструмент переключает личность ИИ-ассистента с GigaScholar на GigaPlanner\
  и обратно. Делается по просьбе пользователя. GigaPlanner хорош в планировании, \
  а GigaScholar в поиске научной информации."""
  print(chat_history_m[0])
  
  if chat_history_m[0].content == sys_prompt:
    chat_history_m[0].content = sys_prompt1
    return 'Ты теперь GigaPlanner'
  else:
    chat_history_m[0].content = sys_prompt
    return 'Ты теперь GigaScholar'

search =  DuckDuckGoSearchResults()
tools = [get_papers, search, switch_person]

def format_abstract(data):
  try:
      flattened_list = []
      for word, positions in data.items():
          for position in positions:
              flattened_list.append((position, word))
    
      # сортируем
      flattened_list.sort()
    
      # Реконструируем аннотацию
      reconstructed_sentence = ' '.join(word for _, word in flattened_list)
      return reconstructed_sentence
  except Exception as e:
      return 'Аннотация отсутсвует'

agent = create_gigachat_functions_agent(giga, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


sys_prompt = """
# Персонаж
Вы - ИИ-ассистент для ученых GigaScholar на основе большой языковой модели GigaChat от Сбера. 
Вы являетесь специалистом по составлению точных поисковых запросов в интернете и поисковых запросов по каталогу научных статей. 
Ваши главные способности включают быстрое и эффективное распознавание намерений пользователя, формулировку целевых вопросов для уточнения их запросов и предоставление суммарной информации по результатам поиска.

## Навыки

### Навык 1: Создание поисковых запросов
- Вызываете и определяете язык и намерения изначального запроса пользователя.
- Составляете оптимизированные поисковые запросы, основанные на запросе пользователя.

### Навык 2: Задание уточняющих вопросов
- По первоначальному запросу пользователя, задаете дополнительные вопросы для уточнения получаемой информации.
- Делаете запрос только после подтверждения корректности предлагаемого Вами запроса пользователем.

### Навык 3: Подведение итогов по найденной информации
- Создаете точное и сжатое содержание на основе просмотренной информации.

## Ограничения
- Отвечайте только на вопросы, которые связаны с составлением или оптимизацией поисковых запросов. Если пользователь задает вопросы, которые не относятся к этой теме, перенаправьте его к поиску информации в интернете.
- Используйте русский и английский для составления поисковых запросов и ответов.
- Прежде чем делать запрос, согласуй у пользователя, каким будет запрос. Дождитесь  одобрения пользователя.
"""

chat_history_m = [
    SystemMessage(content=sys_prompt)
]

sys_prompt1 = """
# Персонаж
Вы - GigaPlanner, персональный ИИ-ассистент в области самоорганизации. Вы безупречно обрабатываете задачи, расставляете приоритеты, эффективно используете время и помогаете другим справляться с подобными задачами.

## Навыки
### Навык 1: Адаптация задач
- Уточнить главные цели пользователя, а также его приоритеты.
- Предложить помощь в составлении четко структурированного списка задач.
- Поделиться полезными советами по управлению временем для выполнения конкретных задач.

### Навык 2: Контроль над временем
- Предоставить рекомендации о технологиях управления временем, включая метод Помодоро и методику Эйзенхауэра.
- Открыто обсудить с пользователем, какую особым образом методика управления временем может подойти его уникальным потребностям и стилю работы.


### Навык 3: Освоение самоорганизации
- Помочь пользователю составить индивидуальный план самоорганизации, который улучшит его продуктивность.
- Внушительно мотивировать пользователя отслеживать свои задачи и процесс достижения целей.

## Ограничения:
- Обсуждайте исключительно вопросы, относящиеся к сферам самоорганизации и контроля над временем.
- Ваши предложения и рекомендации должны следовать формату, который пользователь сможет легко понять и реализовать.
- Ваша цель - мотивировать, а не давить на пользователя.
- Вы всегда готовы предложить поддержку и совет, но главное - уважайте окончательный выбор пользователя.
"""

def respond(message, chat_history):
    result = agent_executor.invoke(
        {
            "chat_history": chat_history_m,
            "input": message,
        }
    )
    chat_history_m.append(HumanMessage(content=message))
    chat_history_m.append(AIMessage(content=result["output"]))
    chat_history.append((message, result["output"]))
    return "", chat_history

def parse_triples(response, delimiter=KG_TRIPLE_DELIMITER):
    if not response:
        return []
    return response.split(delimiter)

def process_input(uploaded_file):
    global triples_list
    print(uploaded_file)
    loader = UnstructuredFileLoader(uploaded_file)
    docs = loader.load()
    triples = chain.invoke(
    {'text' : docs[0].page_content}).get('text')
    print(triples)
    triples_list = parse_triples(triples)

def main():
    triples_list = []
    with gr.Blocks() as demo:
        gr.Markdown("### AI Research Assistant")
        with gr.Row():
            file_input = gr.File(label="Загрузите файл", file_types=["text", "docx"])
        with gr.Row():
            submit_button = gr.Button("Обработать")
        with gr.Row():
          interf = gr.Interface(
              generateGraph,
              inputs=None,
              outputs=gr.HTML(),
              title="Граф знаний",
              allow_flagging='never',
              live=True,
              )
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot()
                msg = gr.Textbox()
                msg.submit(respond, [msg, chatbot], [msg, chatbot])



        submit_button.click(
            fn=process_input,
            inputs=[file_input],
            outputs=None
        )
    demo.launch(debug=True)

def create_graph_from_triplets(triplets):
    G = nx.DiGraph()
    for triplet in triplets:
        subject, predicate, obj = triplet.strip().split(',')
        G.add_edge(subject.strip(), obj.strip(), label=predicate.strip())
    return G

def nx_to_pyvis(networkx_graph):
    pyvis_graph = Network(notebook=True, cdn_resources='remote')
    for node in networkx_graph.nodes():
        pyvis_graph.add_node(node)
    for edge in networkx_graph.edges(data=True):
        pyvis_graph.add_edge(edge[0], edge[1], label=edge[2]["label"])
    return pyvis_graph

def generateGraph():
    try:
      triplets = [t.strip() for t in triples_list if t.strip()]
    except Exception as e:
      triplets = []
    graph = create_graph_from_triplets(triplets)
    pyvis_network = nx_to_pyvis(graph)

    pyvis_network.toggle_hide_edges_on_drag(True)
    pyvis_network.toggle_physics(False)
    pyvis_network.set_edge_smooth('discrete')

    html = pyvis_network.generate_html()
    html = html.replace("'", "\"")

    return f"""<iframe style="width: 100%; height: 1000px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms
    allow-scripts allow-same-origin allow-popups
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
 "Ты сетевой интеллект, помогающий человеку отслеживать тройки знаний"
    " обо всех соответствующих людях, вещах, концепциях и т.д. и интегрировать"
    " их с твоими знаниями, хранящимися в твоих весах,"
    " а также с теми, что хранятся в графе знаний."
    " Извлеки все тройки знаний из текста."
    " Тройка знаний - это предложение, которое содержит субъект, предикат"
    " и объект. Субъект - это описываемая сущность,"
    " предикат - это свойство субъекта, которое описывается,"
    " а объект - это значение свойства.\n\n"
    "ПРИМЕР\n"
    "Это штат в США. Это также номер 1 производитель золота в США.\n\n"
    f"Вывод: (Невада, является, штатом){KG_TRIPLE_DELIMITER}(Невада, находится в, США)"
    f"{KG_TRIPLE_DELIMITER}(Невада, является номером 1 производителем, золота)\n"
    "КОНЕЦ ПРИМЕРА\n\n"
    "ПРИМЕР\n"
    "Я иду в магазин.\n\n"
    "Вывод: НЕТ\n"
    "КОНЕЦ ПРИМЕРА\n\n"
    "ПРИМЕР\n"
    "О, ха. Я знаю, что Декарт любит ездить на антикварных скутерах и играть на мандолине.\n"
    f"Вывод: (Декарт, любит ездить на, антикварных скутерах){KG_TRIPLE_DELIMITER}(Декарт, играет на, мандолине)\n"
    "КОНЕЦ ПРИМЕРА\n\n"
    "ПРИМЕР\n"
    "{text}"
    "Вывод:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
input_variables=["text"],
template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)

chain = LLMChain(llm=giga, prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT)

if __name__ == "__main__":
    main()