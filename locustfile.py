"""Locust baseline: /predict only. No LLM, no rate limit."""

import random
from locust import HttpUser, task, between

TX_IDS = [3433766, 3361860, 2987000, 3128297, 3145309, 3025284,
          3014159, 3092228, 3354110, 3300389]


class PredictUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def predict(self):
        tx = random.choice(TX_IDS)
        self.client.post(f"/predict/{tx}", timeout=10)