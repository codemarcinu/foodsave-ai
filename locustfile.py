from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    wait_time = between(1, 2.5)  # Wait between 1 and 2.5 seconds between tasks

    @task(3)  # More important, frequent path
    def browse_main_page(self):
        self.client.post(
            "/process_query", json={"session_id": "user_1", "query": "what is foodsave"}
        )

    @task(1)  # Less frequent path
    def add_item_to_list(self):
        self.client.post(
            "/process_query",
            json={"session_id": "user_2", "query": "add apple to list"},
        )
