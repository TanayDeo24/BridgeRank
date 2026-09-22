from locust import HttpUser, between, task


class SearchUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task(5)
    def hybrid_search(self):
        self.client.get("/v1/search", params={"q": "vpn cannot resolve internal sites", "top_k": 5})

    @task(1)
    def readiness(self):
        self.client.get("/health/ready")
