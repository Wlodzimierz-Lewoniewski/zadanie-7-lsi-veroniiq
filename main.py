import numpy as np
import math
import re


# Funkcja do przetwarzania tekstu
def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text)


# Funkcja do tworzenia macierzy term-dokument
def compute_document_term_matrix(documents):
    # Tokenizacja dokumentów
    tokenized_documents = []
    for doc in documents:
        doc = preprocess_text(doc.lower())
        tokens = doc.split()
        tokenized_documents.append(tokens)

    # Znajdowanie unikalnych terminów
    unique_tokens = sorted(set(token for tokens in tokenized_documents for token in tokens))
    term_index = {term: i for i, term in enumerate(unique_tokens)}

    # Tworzenie macierzy incydencji
    n = len(documents)
    m = len(unique_tokens)
    C = np.zeros((m, n), dtype=int)

    for j, tokens in enumerate(tokenized_documents):
        for word in tokens:
            if word in term_index:
                C[term_index[word], j] = 1

    return C, unique_tokens, term_index


# Funkcja do przekształcenia zapytania w wektor
def compute_query_vector(query, unique_tokens, term_index):
    query = preprocess_text(query.lower())
    query_tokens = query.split()
    q = np.zeros(len(unique_tokens), dtype=int)
    for word in query_tokens:
        if word in term_index:
            q[term_index[word]] = 1
    return q


# Latent Semantic Indexing (LSI)
def lsi_similarity(documents, query, k):
    # Krok 1: Przygotowanie macierzy term-dokument
    C, unique_tokens, term_index = compute_document_term_matrix(documents)

    # Krok 2: Rozkład SVD
    U, Sigma, VT = np.linalg.svd(C, full_matrices=False)

    # Krok 3: Redukcja do rzędu k
    U_k = U[:, :k]
    Sigma_k = np.diag(Sigma[:k])
    VT_k = VT[:k, :]

    # Krok 4: Obliczanie macierzy zredukowanej
    C_k = np.dot(Sigma_k, VT_k)

    # Krok 5: Obliczanie wektora zapytania w zredukowanej przestrzeni
    q = compute_query_vector(query, unique_tokens, term_index)
    q_k = np.dot(np.linalg.pinv(Sigma_k), np.dot(U_k.T, q))

    # Krok 6: Obliczanie podobieństwa cosinusowego
    similarities = []
    for j in range(C_k.shape[1]):
        doc_vector = C_k[:, j]
        cosine_sim = np.dot(q_k, doc_vector) / (np.linalg.norm(q_k) * np.linalg.norm(doc_vector))
        similarities.append(round(float(cosine_sim), 2))  # Konwersja na typ float

    return similarities

def main(n, documents, query, k):
    result = lsi_similarity(documents, query, k)
    print(result)

if __name__ == "__main__":
    # Wczytywanie danych wejściowych
    n = int(input().strip())  # Liczba dokumentów
    documents = [input().strip() for _ in range(n)]  # Dokumenty
    query = input().strip()  # Zapytanie
    k = int(input().strip())  # Liczba wymiarów

    main(n, documents, query, k)