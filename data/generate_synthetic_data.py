import csv
import random

random.seed(42)

positives = [
    "Great product, highly recommend!",
    "Absolutely loved the quality of these shoes, they are amazing.",
    "Excellent customer service and very fast shipping.",
    "Best purchase I have made this year, totally worth it.",
    "Amazing fit and very comfortable to wear all day.",
    "Super useful product, works exactly as described.",
    "Top notch material, worth every single penny.",
    "Five stars! Very happy with this clothing item.",
    "Exceeded my expectations, will definitely buy again.",
    "Beautiful design and fantastic build quality.",
    "Really good value for money, highly satisfied.",
    "Perfect size, fits like a glove and feels premium.",
    "Awesome customer support, resolved my query in minutes.",
    "The material is incredibly soft and comfortable.",
    "Extremely pleased with this purchase, works like a charm.",
    "Brilliant design, look very stylish and modern.",
    "High quality product, durable and well made.",
    "Outstanding performance, absolutely love it!",
    "Great item, shipping was incredibly fast.",
    "Very comfortable shoes, looks beautiful too."
]

negatives = [
    "Horrible quality, it broke on the first use.",
    "Worst customer service ever, very disappointed with the response.",
    "Terrible fit, the sizes are completely off and tight.",
    "Waste of money, do not recommend this product at all.",
    "Cheap plastic material and very poor design.",
    "Did not work out of the box, returning it immediately.",
    "Awful customer experience, package arrived damaged and open.",
    "Not worth the price, extremely low quality material.",
    "Avoid this product, it is completely useless and slow.",
    "Very bad quality, started falling apart after a week.",
    "Sizing was incorrect and the fabric feels very rough.",
    "Extremely disappointed with this item, useless.",
    "The product look nothing like the pictures, very bad.",
    "Poor customer service, took forever to get a refund.",
    "It broke within two days of normal usage, terrible.",
    "Do not buy! Terrible quality and overpriced.",
    "Very uncomfortable to wear, caused blisters.",
    "Defective item, did not turn on at all.",
    "Awful quality, fabric ripped in the first wash.",
    "Completely unsatisfied, cheap material and bad delivery."
]

neutrals = [
    "Average quality, nothing special but works fine.",
    "It is okay, not great but gets the job done.",
    "Decent shoes, but sizing runs a bit small.",
    "Standard packaging, normal delivery speed.",
    "Okay for the price, but could be much better.",
    "It works as expected, nothing more, nothing less.",
    "Just an ordinary product, does what it says on the box.",
    "Mediocre material, but functional enough for daily use.",
    "So-so experience, neither good nor bad.",
    "Average product, price is a bit high for this quality.",
    "Decent clothing, fabric is fine but color is slightly different.",
    "It is acceptable, nothing to write home about.",
    "The quality is decent, but delivery took a week.",
    "Average customer service, they resolved it but took time.",
    "It fits okay, but the comfort is just average.",
    "Standard product, does the job but lacks premium feel.",
    "Not too bad, but not excellent either.",
    "Fair product, pricing matches the average quality.",
    "It works fine, but design could be improved.",
    "Satisfactory, but I expected slightly better material."
]

reviews = []

for i in range(80):
    review = random.choice(positives)
    if i % 3 == 0:
        review = review + " Highly recommend!"
    elif i % 3 == 1:
        review = "Wow! " + review
    reviews.append((review, "positive"))

for i in range(70):
    review = random.choice(negatives)
    if i % 3 == 0:
        review = review + " Very disappointed."
    elif i % 3 == 1:
        review = "Disaster. " + review
    reviews.append((review, "negative"))

for i in range(50):
    review = random.choice(neutrals)
    if i % 3 == 0:
        review = review + " It is alright."
    elif i % 3 == 1:
        review = "Just ok. " + review
    reviews.append((review, "neutral"))

random.shuffle(reviews)

with open("data/reviews.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Review", "Sentiment"])
    writer.writerows(reviews)

print(f"Generated {len(reviews)} synthetic reviews in data/reviews.csv")
