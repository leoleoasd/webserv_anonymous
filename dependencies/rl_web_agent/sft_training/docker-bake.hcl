group "default" {
  targets = ["nemo"]
}

target "nemo" {
  context = "."
  dockerfile = "Dockerfile"
  args = {
    REPO = "YOUR_ECR_REPO"
    BASE_TAG = "base"
  }
  tags = ["YOUR_ECR_REPO:nemo"]
  platforms = ["linux/amd64"]

  # Cache sources (import)
  cache-from = [
    "type=local,src=./.buildx-cache"
  ]

  # Cache destination (export)
  cache-to = [
    "type=local,dest=./.buildx-cache,mode=max"
  ]
  # cache-to = ["type=registry,ref=YOUR_ECR_REPO:cache,mode=max"]
  # cache-from = ["type=registry,ref=YOUR_ECR_REPO:cache"]
  push = true
}
