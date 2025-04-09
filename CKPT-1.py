import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier

# Baixar recursos do NLTK (apenas se necessário)
# nltk.download('punkt')
# nltk.download('stopwords')

# --- Configurações ---
ARQUIVO_CSV = 'reviews_smartphone.csv'
IDIOMA_STOPWORDS = 'portuguese'
ENCODING_ARQUIVO = 'latin-1'
COLUNA_REVIEWS = 'Reviews'
PORCENTAGEM_TREINO = 0.8

-PALAVRAS_POSITIVAS = [
    'excelente', 'ótima', 'incrível', 'superou', 'adorei', 'melhor', 'rápido', 
    'eficiente', 'recomendo', 'fluido', 'prático', 'vale', 'bom', 'boa', 'gostei', 
    'perfeito', 'maravilhoso', 'satisfeito', 'feliz', 'sucesso', 'ideal', 'top'
]
PALAVRAS_NEGATIVAS = [
    'ruim', 'esquenta', 'arrependi', 'péssima', 'frágil', 'lento', 'bugs', 
    'decepcionante', 'pouca', 'baixo', 'má', 'horrível', 'terrível', 'odeio', 
    'detestei', 'problema', 'falha', 'lixo', 'não funciona', 'travou', 'caro'
]
PALAVRAS_NEUTRAS_EMPATE = ['regular', 'aceitável', 'não é ruim', 'mais ou menos', 'ok', 'mediano']

# --- Funções Auxiliares ---
def classificar_sentimento(texto):
    texto_minusculo = texto.lower()
    count_pos = sum(1 for palavra in PALAVRAS_POSITIVAS if palavra in texto_minusculo)
    count_neg = sum(1 for palavra in PALAVRAS_NEGATIVAS if palavra in texto_minusculo)
    
    if count_pos > count_neg:
        return 'positivo'
    elif count_neg > count_pos:
        return 'negativo'
    else:
        return 'neutro' if any(palavra in texto_minusculo for palavra in PALAVRAS_NEUTRAS_EMPATE) else 'neutro'

def pre_processar(texto, stemmer, stop_words):
    tokens = word_tokenize(texto.lower())
    tokens_filtrados = [palavra for palavra in tokens if palavra.isalpha() and palavra not in stop_words]
    tokens_stemizados = [stemmer.stem(palavra) for palavra in tokens_filtrados]
    return {palavra: True for palavra in tokens_stemizados}

# --- Carregamento e Processamento ---
print(f"Carregando e processando {ARQUIVO_CSV}...")
stop_words = set(stopwords.words(IDIOMA_STOPWORDS))
stemmer = PorterStemmer()
features_sentimentos = []

try:
    with open(ARQUIVO_CSV, 'r', encoding=ENCODING_ARQUIVO) as arquivo:
        leitor_csv = csv.DictReader(arquivo)
        for linha in leitor_csv:
            review = linha[COLUNA_REVIEWS]
            sentimento = classificar_sentimento(review)
            features = pre_processar(review, stemmer, stop_words)
            features_sentimentos.append((features, sentimento))
except FileNotFoundError:
    print(f"Erro: Arquivo {ARQUIVO_CSV} não encontrado.")
    exit()
except KeyError:
    print(f"Erro: Coluna '{COLUNA_REVIEWS}' não encontrada no CSV.")
    exit()

print(f"Total de {len(features_sentimentos)} reviews processados.")

# --- Treinamento e Avaliação ---
if not features_sentimentos:
    print("Nenhum dado para treinar.")
    exit()

# Embaralhar e dividir dados
import random
random.shuffle(features_sentimentos)
limite = int(len(features_sentimentos) * PORCENTAGEM_TREINO)
dados_treino = features_sentimentos[:limite]
dados_teste = features_sentimentos[limite:]

print(f"Treinando classificador com {len(dados_treino)} amostras...")
classificador = NaiveBayesClassifier.train(dados_treino)

# Testar o classificador
print(f"Testando classificador com {len(dados_teste)} amostras...")
acertos = 0
for feature, sentimento_real in dados_teste:
    sentimento_previsto = classificador.classify(feature)
    if sentimento_previsto == sentimento_real:
        acertos += 1

# Calcular e exibir precisão
precisao = (acertos / len(dados_teste)) * 100 if dados_teste else 0
print(f"Precisão do classificador: {precisao:.2f}%")

# Mostrar palavras mais informativas
print("\nPalavras mais informativas:")
classificador.show_most_informative_features(10)

# --- Classificar Novas Reviews --- 
print("\n--- Classificar Novas Reviews ---")
print("Digite uma nova review para classificar (ou 'sair' para terminar):")

while True:
    nova_review = input("> ").lower()
    if nova_review == 'sair':
        break
    
    if not nova_review.strip():
        print("Review vazia. Tente novamente.")
        continue
        
    # Pré-processar a nova review
    features_nova_review = pre_processar(nova_review, stemmer, stop_words)
    
    # Classificar
    sentimento_previsto = classificador.classify(features_nova_review)
    
    print(f"Sentimento previsto: {sentimento_previsto}")
    print("--- Digite outra review ou 'sair' ---")


print("\nProcesso completo!")

