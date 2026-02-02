"""Tests for drone_sim.api.app module.

Tests for FastAPI endpoints:
- GET /health
- POST /config
- POST /step
- GET /state
- GET /render
"""

from __future__ import annotations

import pytest


# =============================================================================
# GET /health Tests
# =============================================================================
class TestHealthEndpoint:
   """Tests for GET /health endpoint."""

   def test_health_returns_ok(self, client):
      """Test /health returns status ok."""
      response = client.get("/health")

      assert response.status_code == 200
      assert response.json()["status"] == "ok"

   def test_health_response_schema(self, client):
      """Test /health response matches HealthResponse schema."""
      response = client.get("/health")

      data = response.json()
      assert "status" in data


# =============================================================================
# POST /config Tests
# =============================================================================
class TestConfigEndpoint:
   """Tests for POST /config endpoint."""

   def test_config_loads_valid_config(self, client, valid_config):
      """Test /config loads a valid configuration."""
      response = client.post("/config", json=valid_config)

      assert response.status_code == 200
      assert response.json()["status"] == "loaded"
      assert response.json()["num_drones"] == 1

   def test_config_multiple_drones(self, client, valid_config):
      """Test /config handles multiple drones."""
      valid_config["drones"].append({
            "drone_id": "d2",
            "physics": "standard",
            "start": [10.0, 10.0, 5.0],
            "waypoints": [],
            "target": [0.0, 0.0, 5.0],
            "radius": 0.2,
            "safety_zone": 1.0
      })

      response = client.post("/config", json=valid_config)

      assert response.status_code == 200
      assert response.json()["num_drones"] == 2

   def test_config_invalid_physics_type_raises(self, client, valid_config):
      """Test /config raises error for invalid physics type."""
      valid_config["physics"][0]["type"] = "nonexistent_physics"

      # The ValueError from registry propagates as an unhandled exception
      # which causes the test client to raise the exception directly
      with pytest.raises(ValueError, match="Unknown physics type"):
         client.post("/config", json=valid_config)

   def test_config_missing_required_field_raises(self, client):
      """Test /config raises error when required field is missing."""
      incomplete_config = {"dt": 0.1, "physics": {"type": "linear_kinematics"}}

      response = client.post("/config", json=incomplete_config)

      assert response.status_code == 422

   def test_config_invalid_drone_start_raises(self, client, valid_config):
      """Test /config raises error for invalid drone start position."""
      valid_config["drones"][0]["start"] = [0.0, 0.0]  # Only 2 elements

      response = client.post("/config", json=valid_config)

      assert response.status_code == 422

   def test_config_replaces_previous_config(self, client, valid_config):
      """Test /config replaces any previously loaded configuration."""
      response1 = client.post("/config", json=valid_config)
      assert response1.status_code == 200

      valid_config["drones"].append({
         "drone_id": "d2",
         "physics": "standard",
         "start": [5.0, 5.0, 5.0],
         "waypoints": [],
         "target": [0.0, 0.0, 5.0]
      })
      response2 = client.post("/config", json=valid_config)

      assert response2.status_code == 200
      assert response2.json()["num_drones"] == 2


# =============================================================================
# POST /step Tests
# =============================================================================
class TestStepEndpoint:
   """Tests for POST /step endpoint."""

   def test_step_without_config_raises_400(self, client):
      """Test /step raises 400 when no config is loaded."""
      response = client.post("/step")

      assert response.status_code == 400
      assert "No scenario configured" in response.json()["detail"]

   def test_step_single_step(self, configured_client):
      """Test /step advances simulation by one step."""
      response = configured_client.post("/step")

      assert response.status_code == 200
      data = response.json()
      assert data["status"] in ["stepped", "infeasible"]
      assert "t" in data
      assert "dt" in data

   def test_step_multiple_steps(self, configured_client):
      """Test /step with n parameter advances multiple steps."""
      response = configured_client.post("/step?n=5")

      assert response.status_code == 200
      data = response.json()
      # Time should have advanced by approximately 5 * dt
      if data["status"] == "stepped":
         assert data["t"] == pytest.approx(5 * data["dt"], rel=0.1)

   def test_step_returns_infeasible_status(self, client):
      """Test /step returns infeasible status when optimization fails."""
      infeasible_config = {
         "dt": 0.1,
         "physics": [{"id": "standard", "type": "linear_kinematics", "params": {}}],
         "controller": {"type": "mpc_agent", "params": {"horizon": 5}},
         "coordinator": {"type": "mpc_central", "params": {"horizon": 5}},
         "drones": [{"drone_id": "d1", "physics": "standard", "start": [-100.0, -100.0, -100.0], "target": [5.0, 5.0, 5.0]}],
         "room": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]}
      }

      client.post("/config", json=infeasible_config)
      response = client.post("/step")

      assert response.status_code == 200
      data = response.json()
      assert data["status"] == "infeasible"
      assert data["detail"] is not None

   def test_step_increments_time(self, configured_client):
      """Test /step increments simulation time."""
      response1 = configured_client.post("/step")
      t1 = response1.json()["t"] if response1.json()["status"] == "stepped" else 0

      response2 = configured_client.post("/step")
      t2 = response2.json()["t"] if response2.json()["status"] == "stepped" else t1

      if response1.json()["status"] == "stepped" and response2.json()["status"] == "stepped":
         assert t2 > t1


# =============================================================================
# GET /state Tests
# =============================================================================
class TestStateEndpoint:
   """Tests for GET /state endpoint."""

   def test_state_without_config_raises_400(self, client):
      """Test /state raises 400 when no config is loaded."""
      response = client.get("/state")

      assert response.status_code == 400
      assert "No scenario configured" in response.json()["detail"]

   def test_state_returns_current_state(self, configured_client):
      """Test /state returns current simulation state."""
      response = configured_client.get("/state")

      assert response.status_code == 200
      data = response.json()
      assert "t" in data
      assert "dt" in data
      assert "room" in data
      assert "drones" in data
      assert "obstacles" in data
      assert "collisions" in data

   def test_state_room_format(self, configured_client):
      """Test /state room has correct format."""
      response = configured_client.get("/state")
      data = response.json()

      assert "min" in data["room"]
      assert "max" in data["room"]
      assert len(data["room"]["min"]) == 3
      assert len(data["room"]["max"]) == 3

   def test_state_drones_format(self, configured_client):
      """Test /state drones have correct format."""
      response = configured_client.get("/state")
      data = response.json()

      assert len(data["drones"]) > 0
      drone = data["drones"][0]
      assert "drone_id" in drone
      assert "x" in drone
      assert len(drone["x"]) == 6
      assert "p_ref" in drone
      assert len(drone["p_ref"]) == 3

   def test_state_updates_after_step(self, configured_client):
      """Test /state reflects changes after step."""
      state1 = configured_client.get("/state").json()
      configured_client.post("/step")
      state2 = configured_client.get("/state").json()

      assert state2["t"] >= state1["t"]


# =============================================================================
# GET /render Tests
# =============================================================================
class TestRenderEndpoint:
   """Tests for GET /render endpoint."""

   def test_render_without_config_raises_400(self, client):
      """Test /render raises 400 when no config is loaded."""
      response = client.get("/render")

      assert response.status_code == 400
      assert "No scenario configured" in response.json()["detail"]

   def test_render_returns_png(self, configured_client):
      """Test /render returns PNG image."""
      response = configured_client.get("/render")

      assert response.status_code == 200
      assert response.headers["content-type"] == "image/png"
      # PNG files start with specific magic bytes
      assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

   def test_render_custom_dimensions(self, configured_client):
      """Test /render accepts custom width/height parameters."""
      response = configured_client.get("/render?width=400&height=300&dpi=72")

      assert response.status_code == 200
      assert response.headers["content-type"] == "image/png"

   def test_render_custom_view_angle(self, configured_client):
      """Test /render accepts custom elev/azim parameters."""
      response = configured_client.get("/render?elev=45&azim=-30")

      assert response.status_code == 200
      assert response.headers["content-type"] == "image/png"

   def test_render_trace_len_parameter(self, configured_client):
      """Test /render accepts trace_len parameter."""
      response = configured_client.get("/render?trace_len=100")

      assert response.status_code == 200

   def test_render_trace_len_zero(self, configured_client):
      """Test /render with trace_len=0."""
      response = configured_client.get("/render?trace_len=0")

      assert response.status_code == 200



# =============================================================================
# GET /swagger Tests
# =============================================================================
class TestSwaggerEndpoint:
   """Tests for GET /swagger endpoint."""

   def test_swagger_returns_html(self, client):
      """Test /swagger returns HTML page."""
      response = client.get("/swagger")

      assert response.status_code == 200
      assert "text/html" in response.headers["content-type"]


# =============================================================================
# Integration Tests
# =============================================================================
class TestApiIntegration:
   """Integration tests for API workflow."""

   def test_full_workflow(self, client, valid_config):
      """Test complete workflow: config -> step -> state -> render."""
      # Load config
      config_response = client.post("/config", json=valid_config)
      assert config_response.status_code == 200

      # Get initial state
      state1 = client.get("/state").json()
      assert state1["t"] == 0.0

      # Step simulation
      step_response = client.post("/step?n=3")
      assert step_response.status_code == 200

      # Get updated state
      state2 = client.get("/state").json()
      if step_response.json()["status"] == "stepped":
         assert state2["t"] > state1["t"]

      # Render visualization
      render_response = client.get("/render")
      assert render_response.status_code == 200
      assert render_response.headers["content-type"] == "image/png"

   def test_reconfigure_resets_simulation(self, client, valid_config):
      """Test reconfiguring resets simulation state."""
      client.post("/config", json=valid_config)
      client.post("/step?n=5")
      state1 = client.get("/state").json()
      assert state1["t"] == 0.5

      client.post("/config", json=valid_config)
      state2 = client.get("/state").json()

      assert state2["t"] == 0.0
