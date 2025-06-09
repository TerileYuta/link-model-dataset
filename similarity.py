from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_cosine_similarity(tokens, target_token):
    tokens = np.array(tokens)
    target_token = np.array(target_token)

    tokens = tokens.reshape(tokens.shape[0], -1)
    target_token = target_token.reshape(1, -1)

    similarity = np.array(cosine_similarity(tokens, target_token))
    similarity = similarity.reshape(-1)

    return np.argpartition(-similarity, 10)[:10]
    