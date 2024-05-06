import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def set_previous_count(count=0):
  redis_client.set("previous_count", count)


def get_previous_count():
  return redis_client.get("previous_count")
