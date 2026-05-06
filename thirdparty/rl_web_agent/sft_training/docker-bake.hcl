group "default" {
  targets = ["nemo"]
}

target "nemo" {
  context = "."
  dockerfile = "Dockerfile"
  args = {
    REPO = "248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland"
    BASE_TAG = "base"
  }
  tags = ["248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland:nemo"]
  platforms = ["linux/amd64"]

  # Cache sources (import)
  cache-from = [
    "type=local,src=./.buildx-cache"
  ]

  # Cache destination (export)
  cache-to = [
    "type=local,dest=./.buildx-cache,mode=max"
  ]
  # cache-to = ["type=registry,ref=248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland:cache,mode=max"]
  # cache-from = ["type=registry,ref=248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland:cache"]
  push = true
}
