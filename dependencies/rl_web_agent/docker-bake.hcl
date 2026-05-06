group "default" {
  targets = ["sglang"]
}

target "sglang" {
  context = "."
  dockerfile = "Dockerfile"
  args = {
    ROLLOUT_ENGINE = "sglang"
    REPO = "YOUR_ECR_REPO"
    BASE_TAG = "base"
  }
  tags = ["YOUR_ECR_REPO:verl_sglang"]
  platforms = ["linux/amd64"]
  cache-to = ["type=registry,ref=YOUR_ECR_REPO:cache,mode=max"]
  cache-from = ["type=registry,ref=YOUR_ECR_REPO:cache"]
  push = true
}
