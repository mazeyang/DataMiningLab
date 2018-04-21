import numpy as np
import lda
import lda.datasets

# document-term matrix
X = lda.datasets.load_reuters()
print("type(X): {}".format(type(X)))
print("shape: {}\n".format(X.shape))
print(X[:5, :5])

# the vocab
vocab = lda.datasets.load_reuters_vocab()
print("type(vocab): {}".format(type(vocab)))
print("len(vocab): {}\n".format(len(vocab)))
print(vocab[:5])

# titles for each story
titles = lda.datasets.load_reuters_titles()
print("type(titles): {}".format(type(titles)))
print("len(titles): {}\n".format(len(titles)))
print(titles[:5])

print('--------------------------------------')

# X[0,3117] is the number of times that word 3117 occurs in document 0
doc_id = 0
word_id = 3117
print("doc id: {} word id: {}".format(doc_id, word_id))
print("-- count: {}".format(X[doc_id, word_id]))
print("-- word : {}".format(vocab[word_id]))
print("-- doc  : {}".format(titles[doc_id]))

print('--------------------------------------')

model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
model.fit(X)          # model.fit_transform(X) is also available

topic_word = model.topic_word_
print("type(topic_word): {}".format(type(topic_word)))
print("shape: {}".format(topic_word.shape))
print(vocab[:3])
print(topic_word[:, :3])

for n in range(5):
    sum_pr = sum(topic_word[n,:])
    print("topic: {} sum: {}".format(n, sum_pr))

n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

print('--------------------------------------')

# The most possible topics of top 10 texts.
doc_topic = model.doc_topic_
print("type(doc_topic): {}".format(type(doc_topic)))
print("shape: {}".format(doc_topic.shape))
for n in range(10):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}".format(n, topic_most_pr))
