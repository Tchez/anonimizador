import spacy
import re
import streamlit as st
from spacy import displacy
from spacy.util import filter_spans  # Função para filtrar spans sobrepostos

# Carregar o modelo de linguagem em português
nlp = spacy.load("pt_core_news_sm")


def mask_entities_and_contacts(text):
    """
    Esta função anonimiza informações sensíveis em um texto fornecido.

    Args:
        text (str): O texto a ser processado.

    Returns:
        tuple: Texto anonimizado, lista de entidades encontradas e o objeto doc do spaCy.
    """

    # Processar o texto com o modelo do spaCy
    doc = nlp(text)

    # Lista para armazenar os spans (início e fim) das partes a serem mascaradas
    spans = []

    # Lista para armazenar as entidades identificadas (texto e label)
    entities_found = []

    # Iterar sobre as entidades reconhecidas pelo modelo do spaCy
    for ent in doc.ents:
        if ent.label_ in ["PER", "ORG", "LOC", "GPE"]:  # Labels em português
            spans.append(
                (ent.start_char, ent.end_char)
            )  # Adicionar posições de início e fim
            entities_found.append(
                {'text': ent.text, 'label': ent.label_}
            )  # Salvar entidade

    # Definir padrões para CPF, CNPJ, telefones e emails
    patterns = [
        # CPF: Torna a palavra "CPF" opcional
        {
            'label': 'CPF',
            'pattern': r'\b(?:CPF\s*)?\d{3}\.\d{3}\.\d{3}-\d{2}\b',
        },
        # CNPJ: Torna a palavra "CNPJ" opcional e corrige o formato
        {
            'label': 'CNPJ',
            'pattern': r'\b(?:CNPJ\s*)?\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b',
        },
        # TELEFONE: Torna a palavra "telefone" opcional
        {
            'label': 'TELEFONE',
            'pattern': r'\b(?:telefone\s*)?(\(?\d{2}\)?\s*)?9?\d{4}-\d{4}\b',
        },
        # EMAIL: Padrão para emails
        {'label': 'EMAIL', 'pattern': r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b'},
    ]

    # Criar uma lista com as entidades já reconhecidas pelo spaCy
    new_ents = list(doc.ents)

    # Iterar sobre os padrões definidos
    for item in patterns:
        # Usar expressões regulares para encontrar correspondências no texto
        for match in re.finditer(item['pattern'], text, flags=re.IGNORECASE):
            # Criar uma nova entidade usando os índices da correspondência
            span = doc.char_span(
                match.start(), match.end(), label=item['label']
            )
            if span is not None:
                new_ents.append(span)  # Adicionar a nova entidade à lista
                entities_found.append(
                    {'text': span.text, 'label': span.label_}
                )
                spans.append(
                    (span.start_char, span.end_char)
                )  # Adicionar posições para mascarar

    # Filtrar entidades sobrepostas usando a função filter_spans do spaCy
    filtered_ents = filter_spans(new_ents)

    # Atualizar as entidades do documento com as entidades filtradas
    doc.ents = filtered_ents

    # Ordenar os spans por posição inicial
    spans = sorted(spans, key=lambda x: x[0])

    # Combinar spans sobrepostos para evitar mascaramento duplicado
    merged_spans = []
    for span in spans:
        if not merged_spans:
            merged_spans.append(span)
        else:
            last_span = merged_spans[-1]
            if span[0] <= last_span[1]:
                # Se houver sobreposição, combinar os spans
                new_span = (last_span[0], max(last_span[1], span[1]))
                merged_spans[-1] = new_span
            else:
                merged_spans.append(span)

    # Construir o texto anonimizado substituindo as partes sensíveis por '********'
    masked_text = ''
    last_idx = 0
    for start, end in merged_spans:
        masked_text += text[last_idx:start]  # Texto antes da entidade
        masked_text += '********'  # Máscara
        last_idx = end  # Atualizar o índice
    masked_text += text[last_idx:]  # Adicionar o restante do texto

    return masked_text, entities_found, doc


def main():
    """
    Função principal que executa o aplicativo Streamlit.
    """
    st.title("Anonimizador de Informações Sensíveis")
    st.write(
        "Este aplicativo anonimiza informações pessoais em texto e mostra quais informações foram identificadas."
    )

    # Caixa de texto para o usuário inserir o texto
    text_input = st.text_area("Insira o texto aqui", height=200)

    # Botão para acionar a anonimização
    if st.button("Anonimizar"):
        if text_input:
            # Chamar a função de anonimização
            masked_text, entities_found, doc = mask_entities_and_contacts(
                text_input
            )
            st.subheader("Texto Anonimizado")
            st.write(masked_text)

            # Mostrar as entidades identificadas
            if entities_found:
                st.subheader("Informações Sensíveis Identificadas")
                for entity in entities_found:
                    st.write(f"**{entity['label']}**: {entity['text']}")
            else:
                st.write("Nenhuma informação sensível identificada.")

            # Visualização das entidades no texto original
            st.subheader("Visualização das Entidades Identificadas")
            colors = {
                "PER": "yellow",
                "ORG": "blue",
                "LOC": "green",
                "CPF": "red",
                "CNPJ": "orange",
                "TELEFONE": "purple",
                "EMAIL": "pink",
            }
            options = {
                "ents": [
                    "PER",
                    "ORG",
                    "LOC",
                    "CPF",
                    "CNPJ",
                    "TELEFONE",
                    "EMAIL",
                ],
                "colors": colors,
            }
            # Gerar a visualização usando displacy
            html = displacy.render(doc, style="ent", options=options)
            # Ajustar o HTML para exibir corretamente no Streamlit
            html = html.replace("\n", " ")
            st.write(
                f'<div style="overflow-x: auto; padding: 1em;">{html}</div>',
                unsafe_allow_html=True,
            )

        else:
            st.warning("Por favor, insira o texto para anonimização.")


if __name__ == "__main__":
    main()
