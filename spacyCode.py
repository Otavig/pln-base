import spacy

class SpacyClass:
    def __init__(self, model="pt_core_news_sm"):
        self.nlp = spacy.load(model)

    def segmentation(self, texto, imprimir=False):
        doc = self.nlp(texto)
        resultado = [sent.text for sent in doc.sents]

        if imprimir:
            for sentenca in resultado:
                print(sentenca)

        return resultado

    def tokenization(self, texto, imprimir=False):
        doc = self.nlp(texto)
        resultado = [token.text for token in doc if not token.is_punct]

        if imprimir:
            print(resultado)

        return resultado

    def labeling(self, texto, imprimir=False):
        doc = self.nlp(texto)
        resultado = [
            {
                "token": token.text,
                "pos": token.pos_,
                "descricao": token.pos_,
                "tag": token.tag_
            }
            for token in doc if not token.is_punct
        ]

        if imprimir:
            for item in resultado:
                print(
                    f"Token: {item['token']} | "
                    f"POS: {item['pos']} | Tag: {item['tag']}"
                )

        return resultado

    def stemming(self, palavra, sufixos, imprimir=False):
        palavra = palavra.lower()
        resultado = palavra

        for sufixo in sorted(sufixos, key=len, reverse=True):
            if palavra.endswith(sufixo):
                resultado = palavra[:-len(sufixo)]
                break

        if imprimir:
            print(f"Palavra: {palavra} | Stem: {resultado}")

        return resultado

    def lemmatization(self, texto, imprimir=False):
        doc = self.nlp(texto)
        resultado = [
            {"token": token.text, "lema": token.lemma_}
            for token in doc if token.is_alpha
        ]

        if imprimir:
            for item in resultado:
                print(f"{item['token']} - {item['lema']}")

        return resultado

    def stop_words(self, texto, imprimir=False, remover=False):
        doc = self.nlp(texto)

        if remover:
            resultado = [
                token.text for token in doc
                if token.is_alpha and not token.is_stop
            ]
        else:
            resultado = [
                token.text for token in doc
                if token.is_alpha and token.is_stop
            ]

        if imprimir:
            print(resultado if not remover else " ".join(resultado))

        return resultado

    def named_entities(self, texto, imprimir=False):
        doc = self.nlp(texto)
        resultado = [
            {"nome": entidade.text, "tipo": entidade.label_}
            for entidade in doc.ents
        ]

        if imprimir:
            for entidade in resultado:
                print(f"Nome: {entidade['nome']} | Tipo: {entidade['tipo']}")

        return resultado

    def proper_nouns(self, texto, imprimir=False):
        doc = self.nlp(texto)
        resultado = [
            {"token": token.text, "lema": token.lemma_}
            for token in doc if token.pos_ == "PROPN"
        ]

        if imprimir:
            for nome in resultado:
                print(f"Nome próprio: {nome['token']} | Lema: {nome['lema']}")

        return resultado


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


print("\n-Tokenization-")
resultado_tokens = spacy_class.tokenization(texto, imprimir=True)

print("\n-Labeling-")
resultado_labeling = spacy_class.labeling(texto, imprimir=True)

print("\n-Stop Words-")
resultado_stop_words = spacy_class.stop_words(texto, imprimir=True)

print("\n-Texto sem Stop Words-")
texto_filtrado = spacy_class.stop_words(
    texto, imprimir=True, remover=True
)

print("\n-Segmentação-")
resultado_segmentacao = spacy_class.segmentation(texto, imprimir=True)

print("\n-Stemming-")
sufixos = ["mente", "ções", "ção", "ando", "endo", "idos", "ido", "os", "as", "s"]
print(spacy_class.stemming("rapidamente", sufixos, imprimir=True))
print(spacy_class.stemming("estudando", sufixos, imprimir=True))
print(spacy_class.stemming("casas", sufixos, imprimir=True))

print("\n-Lemmatization-")
resultado_lemas = spacy_class.lemmatization(texto, imprimir=True)

print("\n-Named Entities-")
resultado_entidades = spacy_class.named_entities(texto, imprimir=True)

print("\n-Proper Nouns-")
resultado_nomes = spacy_class.proper_nouns(texto, imprimir=True)
