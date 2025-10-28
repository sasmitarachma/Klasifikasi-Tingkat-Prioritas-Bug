import requests
import re

class GithubScanner:
    def __init__(self, repo_url):
        self.repo_url = repo_url
        self.api_base = "https://api.github.com/repos/"
        self.owner, self.repo = self.parse_repo_url(repo_url)

    def parse_repo_url(self, repo_url):
        parts = repo_url.rstrip('/').split('/')
        if len(parts) < 5:
            raise ValueError("Invalid GitHub repository URL")
        return parts[-2], parts[-1]
    
    def get_issues(self, state='open', per_page=100):
        url = f"{self.api_base}{self.owner}/{self.repo}/issues?state={state}"
        params = {'state': state, 'per_page': per_page}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch issues: {response.status_code} - {response.text}")
    
    def extract_bug_summaries(self, issues_json):
        summaries = []
        for issue in issues_json:
            # Abaikan pull request (PR bukan issue)
            if 'pull_request' in issue:
                continue
            title = issue.get('title', '').strip()
            if title:
                summaries.append(title)
        return summaries


    def scan_repo(self):
        issues_json = self.get_issues()
        return self.extract_bug_summaries(issues_json)