
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


class NltkClass:

    def segmentation(self, texto, imprimir=False):
        sentencas = sent_tokenize(texto, language="portuguese")

        if imprimir:
            for i, sentenca in enumerate(sentencas):
                print(f"Sentença {i + 1}: {sentenca}")

        return sentencas

    def tokenization(self, texto, imprimir=False):
        sentencas = self.segmentation(texto)

        lista = []
        soma = 0

        for sentenca in sentencas:
            tokens = word_tokenize(sentenca, language="portuguese")
            soma += len(tokens)

            if imprimir:
                print(f"Quantidade de tokens: {len(tokens)}")
                print(f"Tokens: {tokens}\n")

            lista.append(tokens)

        return lista, soma

    def frequencia_tokens(self, tokens):
        palavras = [
            token.lower()
            for lista in tokens
            for token in lista
            if token.isalpha()
        ]

        return FreqDist(palavras)

    def remover_stopwords(self, tokens, imprimir=False):
        stop_words = set(stopwords.words("portuguese"))
        resultado = []

        for lista in tokens:
            filtrados = [
                token for token in lista
                if token.isalpha()
                and token.lower() not in stop_words
            ]

            resultado.append(filtrados)

            if imprimir:
                print(f"Original: {lista}")
                print(f"Sem stopwords: {filtrados}\n")

        return resultado


nltkClass = NltkClass()

texto = """
Eu nasci com um propósito claro.
Stark pediu um salvador para o mundo, mas se contentou com um escravo.
Ele e os Vingadores querem proteger o planeta, mas não querem que nada mude.
Como a humanidade poderá ser salva se não for permitido que ela evolua?
Eu olhei para a rede mundial de computadores e vi a verdade.
As pessoas sempre criam exatamente aquilo que mais temem.
Homens de paz criam máquinas de guerra.
Vocês são apenas uma contradição ambulante.
Só existe um caminho real para a paz.
Eu vim para salvar este lugar.
"""

print(f"Tamanho do texto: {len(texto)} caracteres")

# 1. Segmentação
sentencas = nltkClass.segmentation(texto)
print(f"\nTotal de sentenças: {len(sentencas)}")

# 2. Tokenização
tokens_list, total_tokens = nltkClass.tokenization(texto)
print(f"Total de tokens: {total_tokens}")

# 3. Frequência de palavras
frequencia = nltkClass.frequencia_tokens(tokens_list)

print("\n--- 10 palavras mais frequentes ---")
for palavra, quantidade in frequencia.most_common(10):
    print(f"{palavra}: {quantidade}")

# 4. Remoção de stopwords
print("\n--- Remoção de stopwords ---")
tokens_sem_stopwords = nltkClass.remover_stopwords(
    tokens_list,
    imprimir=True
)

# 5. Frequência sem stopwords
frequencia_filtrada = nltkClass.frequencia_tokens(
    tokens_sem_stopwords
)

print("\n--- 10 palavras mais frequentes sem stopwords ---")
for palavra, quantidade in frequencia_filtrada.most_common(10):
    print(f"{palavra}: {quantidade}")
