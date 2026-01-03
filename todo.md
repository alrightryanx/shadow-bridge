# Shadow Bridge Web Dashboard - Feature Parity TODO

## Ecosystem Gaps to Fix
Features that Android app has but web dashboard lacks (or has API but no UI).

---

## Priority 1: Approvals UI (Safety-Critical)
- [ ] Create `web/templates/approvals.html`
- [ ] Add `/approvals` route to `web/app.py`
- [ ] Add nav link to `web/templates/base.html`

**Features:**
- Pending decisions list with approve/deny buttons
- Decision details (agent, action, context, risk level)
- Pause/Resume all agents
- Kill switch (emergency stop)
- Decision history log

**APIs (exist):**
- `GET /api/override/decisions`
- `POST /api/override/decisions/<id>/approve`
- `POST /api/override/decisions/<id>/deny`
- `POST /api/override/pause` / `resume`
- `POST /api/override/kill-switch`

---

## Priority 2: Usage/Cost Analytics
- [ ] Enhance `web/templates/analytics.html` with full usage data
- [ ] Add chart.js or similar for visualizations
- [ ] Add missing API calls to `web/static/js/api.js`

**Features:**
- Token usage over time (chart)
- Cost breakdown by backend (Claude, GPT, Gemini)
- Usage by feature (chat, automations, image gen)
- Daily/weekly/monthly views
- Export usage report

**APIs (exist):**
- `GET /api/analytics/tokens`
- `GET /api/analytics/categories`
- `GET /api/usage/stats`
- `GET /api/usage/backend`
- `GET /api/usage/timeline`

---

## Priority 3: Agent Registry Enhancement
- [ ] Add registry section to `web/templates/agents.html`

**Features:**
- Registered agents list with capabilities
- Agent discovery (find available agents)
- Agent health/heartbeat status
- Create agent teams visually

**APIs (exist):**
- `GET /api/registry/agents`
- `GET /api/registry/discover`
- `POST /api/registry/create-team`
- `GET /api/registry/capabilities`

---

## Priority 4: Permission Policy Editor
- [ ] Create `web/templates/permissions.html`
- [ ] Add `/permissions` route

**Features:**
- View permission rules
- Create/edit rules visually
- Set trust levels per agent
- View permission history

**APIs (exist):**
- `GET /api/permissions/rules`
- `POST /api/permissions/trust`
- `GET /api/permissions/history`

---

## Priority 5: Favorites Page
- [ ] Create `web/templates/favorites.html`
- [ ] Add `/favorites` route

**Features:**
- Combined view of favorite projects + notes
- Quick access from sidebar
- Remove from favorites

**APIs (exist):**
- `GET /api/favorites`
- `POST /api/projects/<id>/favorite`
- `POST /api/notes/<id>/favorite`

---

## Not Needed in Web Dashboard
- Terminal/SSH shell (PC already has terminal)
- File browser (PC already has file explorer)
- Voice input (PC has keyboard)
- On-device LLM (PC runs full models)

---

## After Completion
- [ ] Increment version to 1.026
- [ ] Rebuild EXE with `pyinstaller ShadowBridge.spec`
- [ ] Commit and push
