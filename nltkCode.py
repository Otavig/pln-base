import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import gutenberg, stopwords
from nltk import FreqDist
from nltk.stem import RSLPStemmer, WordNetLemmatizer


nltk.download("punkt_tab")
nltk.download("gutenberg")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("rslp")
nltk.download("wordnet")
nltk.download("omw-1.4")


class NltkClass:

    def __init__(self):
        self.stemmer = RSLPStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    # Aula 1 - Segmentação
    def segmentation(self, texto, imprimir=False):
        sentencas = sent_tokenize(texto)

        if imprimir:
            for i, sentenca in enumerate(sentencas):
                print(f"Sentença {i + 1}: {sentenca}")

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
                print(f"Tokens: {tokens}\n")

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
        resultado = []

        for lista in tokens:
            tokens_limpos = [
                token
                for token in lista
                if token.isalpha()
                and token.lower() not in self.stop_words
            ]

            resultado.append(tokens_limpos)

            if imprimir:
                print(tokens_limpos)

        return resultado

    # Aula 2 - Stemização
    def stemm(self, palavra):
        return self.stemmer.stem(palavra)

    # Aula 2 - Lematização
    def lemm(self, palavra):
        return self.lemmatizer.lemmatize(palavra)

    # Aula 2 - Identificação de nomes próprios
    def nomes_proprios(self, tokens, imprimir=False):
        resultado = []

        for lista in tokens:
            tags = nltk.pos_tag(lista)

            nomes = [
                (palavra, classe)
                for palavra, classe in tags
                if classe == "NNP" or classe == "NNPS"
            ]

            resultado.append(nomes)

            if imprimir:
                for palavra, classe in nomes:
                    print(f"{palavra} --> {classe}")

        return resultado


nltkClass = NltkClass()


# Aula 1 - Segmentação e Tokenização

texto = gutenberg.raw("carroll-alice.txt")

print("Quantidade de caracteres:", len(texto))
print(texto[:1000])

# Segmentação
sentencas = nltkClass.segmentation(texto)

print("\nQuantidade de sentenças:", len(sentencas))

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


# Aula 2 - Rotulação e Stopwords

texto = """
The data scientist is studying Natural Language Processing.
He wants to build an intelligent chatbot.
Alice and Bob are learning Python.
"""

# Segmentação
frases = nltkClass.segmentation(texto)

print("\nFRASES:")

for frase in frases:
    print(frase)


# Tokenização
list_tokens, total_tokens = nltkClass.tokenization(texto)

print("\nTOKENS:")

for lista in list_tokens:
    print([token.lower() for token in lista])

print("Total de tokens:", total_tokens)


# Remoção de stopwords
print("\nTOKENS SEM STOPWORDS:")

tokens_limpos = nltkClass.remover_stopwords(
    list_tokens,
    imprimir=True
)


# Rotulação gramatical
print("\nROTULAÇÃO GRAMATICAL:")

tokens_rotulados = nltkClass.rotulacao(
    tokens_limpos,
    imprimir=True
)


# Identificação de nomes próprios
print("\nNOMES PRÓPRIOS:")

nomes = nltkClass.nomes_proprios(
    list_tokens,
    imprimir=True
)


# Stemização - português
print("\nSTEMIZAÇÃO:")

palavras = ["estudando", "estudar", "estudos", "brincando"]

for palavra in palavras:
    print(f"{palavra} --> {nltkClass.stemm(palavra)}")


# Lematização - inglês
print("\nLEMATIZAÇÃO:")

palavras = ["cars", "running", "studies", "rocks"]

for palavra in palavras:
    print(f"{palavra} --> {nltkClass.lemm(palavra)}")