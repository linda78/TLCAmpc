variable "hcloud_token" {
  description = "Hetzner Cloud API token"
  type        = string
  sensitive   = true
}

variable "worker_count" {
  description = "Number of worker servers to provision"
  type        = number
  default     = 3
}

variable "worker_server_type" {
  description = "Hetzner server type for workers"
  type        = string
  default     = "cpx42"
}

variable "db_server_type" {
  description = "Hetzner server type for the DB server"
  type        = string
  default     = "cpx22"
}

variable "deploy_key_private" {
  description = "SSH private deploy key for git clone (read-only access to repo)"
  type        = string
  sensitive   = true
}

variable "git_repo_url" {
  description = "SSH URL of the git repository"
  type        = string
  default     = "git@github.com:linda78/TLCAmpc.git"
}

variable "git_branch" {
  description = "Git branch to check out on workers"
  type        = string
  default     = "main"
}

variable "db_password" {
  description = "PostgreSQL password for the dronesim user"
  type        = string
  sensitive   = true
}

variable "location" {
  description = "Hetzner datacenter location"
  type        = string
  default     = "fsn1"
}

# Scenario parameters passed to the worker systemd service
variable "scenario_v_max" {
  type    = number
  default = 2.5
}

variable "scenario_u_max" {
  type    = number
  default = 3.0
}

variable "scenario_room_size" {
  type    = number
  default = 8.0
}

variable "scenario_static_safety_zone" {
  type    = number
  default = 1.52
}

variable "scenario_adaptive_safety_zone" {
  type    = number
  default = 1.0
}

variable "scenario_r_min" {
  type    = number
  default = 0.4
}
