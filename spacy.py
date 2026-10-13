import spacy

class SpacyClass:
    def __init__(self):
        self.nlp = spacy.load("pt_core_news_sm")

    def segmentation(self, texto): # dividir o texto em setenças (sujeito e predicato)
        doc = self.nlp(texto)

        setencas = doc.sents # Generator - lista
        for sent in setencas:
            print(sent.text)

    def tokenization(self, texto):
        doc = self.nlp(texto)

        for token in doc:
            if not token.is_punct:
                print(f"Token {token.text} | Rotulo: {token.tag_}")

    def stemming(self, palavra, sufixos): # sufixos ametos, imentos, amento, imento, adoras, adores, antes, ando, endo, indo, ados, edos, idos
        palavra = palavra.lower()

        for sufixo in sufixos:
            if palavra.endswith(sufixo):
                return palavra[:-len(sufixo)]

        return palavra

    def lemmatization(self, texto):
      doc = self.nlp(texto)

      for token in doc:
        if token.is_alpha:
          print(f"{token.text} - {token.lemma_}")


spacy_class = SpacyClass()

texto = "Os programadores estavam desenvolvendo aplicações. Eles trabalhavam bastante."

print("--- Segmentação ---")
spacy_class.segmentation(texto)

print("\n--- Tokenização ---")
spacy_class.tokenization(texto)

print("\n--- Stemming ---")
sufixos = [
    "amentos", "imentos", "amento", "imento",
    "ando", "endo", "indo", "ados", "idos"
]

print(spacy_class.stemming("desenvolvendo", sufixos))

print("\n--- Lemmatization ---")
spacy_class.lemmatization(texto)
