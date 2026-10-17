"""Everyday topics. Prompts are "Write two sentences about <topic>." Three separate sets so nothing is tested on what it was tuned or trained on."""
TRAIN = ["a bakery", "the beach", "a library", "a snowy morning", "a football match", "a garden", "a bus stop", "a wedding",
         "a museum", "a lake", "a market", "a kitchen", "the mountains", "a bookshop", "a summer holiday", "a playground",
         "the moon", "a river", "a picnic", "a village", "a phone call", "an airport", "a forest", "a classroom"]
TUNE = ["coffee", "the sea", "Mondays", "a train journey", "autumn", "a birthday party", "the park", "a cat",
        "rain", "the city at night", "an old house", "the first day of school"]
TEST = ["a dog", "breakfast", "the weekend", "a bicycle", "the zoo", "a sunset", "a concert", "a farm", "winter",
        "a letter", "a hospital", "a train station", "a new job", "an island", "a family dinner", "a thunderstorm",
        "a road trip", "a swimming pool", "spring", "a lighthouse", "a café", "a school bus", "a festival", "the stars"]


def prompts(topics, mood=""):
    """mood="" gives a plain request; mood="sad" or "happy" asks for that mood."""
    return ["Write two %ssentences about %s." % (mood + " " if mood else "", t) for t in topics]
