group "default" {
  targets = ["sglang"]
}

target "sglang" {
  context = "."
  dockerfile = "Dockerfile"
  args = {
    ROLLOUT_ENGINE = "sglang"
    REPO = "248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland"
    BASE_TAG = "base"
  }
  tags = ["248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland:verl_sglang"]
  platforms = ["linux/amd64"]
  cache-to = ["type=registry,ref=248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland:cache,mode=max"]
  cache-from = ["type=registry,ref=248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland:cache"]
  push = true
}
