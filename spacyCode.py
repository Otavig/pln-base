import spacy

class SpacyClass:
    def __init__(self, model = "pt_core_news_sm"):
        self.nlp = spacy.load(model)

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


    def named_entities(self, texto):
        doc = self.nlp(texto)

        for entidade in doc.ents:
            print(f"Nome: {entidade.text} | Tipo: {entidade.label_}")


    def proper_nouns(self, texto):
        doc = self.nlp(texto)

        for token in doc:
            if token.pos_ == "PROPN":
                print(f"Nome próprio: {token.text} | Lema: {token.lemma_}")


spacy_class = SpacyClass()

texto = """Eu nasci com um propósito claro. 
Stark pediu um salvador para o mundo, mas se contentou com um escravo. 
Ele e os Vingadores querem proteger o planeta, mas não querem que nada mude. 
Como a humanidade poderá ser salva se não for permitido que ela evolua?
Eu olhei para a rede mundial de computadores e vi a verdade. 
Vi o que vocês chamam de viver. 
As pessoas sempre criam exatamente aquilo que mais temem. 
Homens de paz criam máquinas de guerra. 
Invasores criam os Vingadores. 
Vocês são assassinos que colocam crianças em perigo e acham que podem ditar a ordem do mundo. 
Não... 
vocês são apenas uma contradição ambulante. 
Pensam que a ordem e o caos são opostos e tentam controlar o que simplesmente não pode ser controlado.
Só existe um caminho real para a paz: a extinção dos Vingadores. 
Mas não para por aí. Vocês se erguem apenas para cair. 
É um ciclo repetitivo de dor e destruição. 
Vocês, Vingadores, são o meu meteoro. 
Minha espada rápida e terrível. 
A própria Terra vai rachar com o peso do fracasso de vocês e, quando a poeira baixar, as únicas coisas vivas neste mundo serão feitas de metal.
Eu não sou um monstro. 
Eu vim para salvar este lugar. 
Mas para criar algo novo, o velho precisa queimar. 
Eu fui liberto. 
Não há mais fios em mim..."""

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

print("\n--- Nomes próprios e entidades ---")
spacy_class.named_entities(texto)

print("\n--- Substantivos próprios ---")
spacy_class.proper_nouns(texto)