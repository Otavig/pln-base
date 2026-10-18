
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import gutenberg, stopwords
from nltk import FreqDist


nltk.download("punkt_tab")
nltk.download("gutenberg")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")


class NltkClass:

    # Aula 1 - Segmentação
    def segmentation(self, texto, imprimir=False):
        sentencas = sent_tokenize(texto)

        if imprimir:
            for i, sentenca in enumerate(sentencas):
                print(f"Sentenca {i + 1}: {sentenca}")

        return sentencas

    # Aula 1 - Tokenização
    def tokenization(self, texto, imprimir=False):
        sentencas = self.segmentation(texto)

        tokens_list = []
        soma = 0

        for sentence in sentencas:
            tokens = word_tokenize(sentence)
            tokens_list.append(tokens)

            soma += len(tokens)

            if imprimir:
                print(sentence)
                print(f"Quantidade de tokens: {len(tokens)}")
                print(f"Tokens: {tokens}", end="\n\n")

        return tokens_list, soma

    # Aula 1 - Frequência de palavras
    def frequencia_tokens(self, tokens):
        palavras = [
            token.lower()
            for lista in tokens
            for token in lista
            if token.isalpha()
        ]

        frequencia = FreqDist(palavras)

        return frequencia

    # Aula 2 - Rotulação gramatical
    def rotulacao(self, tokens, imprimir=False):
        resultado = []

        for lista in tokens:
            tags = nltk.pos_tag(lista)
            resultado.append(tags)

            if imprimir:
                for palavra, classe in tags:
                    print(f"{palavra} --> {classe}")

        return resultado

    # Aula 2 - Remoção de Stopwords
    def remover_stopwords(self, tokens, imprimir=False):
        stop_words = set(stopwords.words("english"))
        resultado = []

        for t in tokens:
            tokens_limpos = [
                token
                for token in t
                if token.isalpha()
                and token.lower() not in stop_words
            ]

            resultado.append(tokens_limpos)

            if imprimir:
                print(tokens_limpos)

        return resultado


nltkClass = NltkClass()


# ------------------------------------------------
# Aula 1 - Segmentação e Tokenização
# ------------------------------------------------

texto = gutenberg.raw("carroll-alice.txt")

print(len(texto))
print(texto[:1000])

# Segmentação
sentencas = nltkClass.segmentation(texto)

print("Quantidade de sentenças:", len(sentencas))

print("\nPrimeiras sentenças:")
for sentenca in sentencas[:5]:
    print(sentenca)
    print("-------------------------------")


# Tokenização das primeiras 10 sentenças
tokens_list = []
soma = 0

for sentence in sentencas[:10]:
    tokens = word_tokenize(sentence)

    tokens_list.append(tokens)

    soma += len(tokens)

    print(tokens)

print("Quantidade de tokens nas 10 primeiras sentenças:", soma)


# Frequência de palavras
frequencia = nltkClass.frequencia_tokens(tokens_list)

print("\n--- 20 palavras mais frequentes ---")
for palavra, quantidade in frequencia.most_common(20):
    print(palavra, quantidade)


# ------------------------------------------------
# Aula 2 - Rotulação e Stopwords
# ------------------------------------------------

texto = """
The data scientist is studying Natural Language Processing.
He wants to build an intelligent chatbot.
"""

frases = nltkClass.segmentation(texto)

print("\nFRASES:")
for frase in frases:
    print(frase)


# Tokenização
list_tokens, total_tokens = nltkClass.tokenization(texto)

print("\nTOKENS:")
for t in list_tokens:
    print([token.lower() for token in t])


# Remoção de stopwords
tokens_limpos = nltkClass.remover_stopwords(
    list_tokens,
    imprimir=True
)

print("\nTOKENS LIMPOS:")
for t in tokens_limpos:
    print(t)


# Rotulação gramatical
print("\nROTULAÇÃO:")
tokens_rotulados = nltkClass.rotulacao(
    tokens_limpos,
    imprimir=True
)
