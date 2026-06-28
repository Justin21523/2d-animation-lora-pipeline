---
schema_version: "inazuma_character_v1.0"
doc_type: "character_profile"
character_id: "inamori_asuto"
canonical_name: "Inamori Asuto"
jp_name: "稲森 明日人"
english_name: "Sonny Wright"
gender: "male"
primary_positions: ['forward', 'midfielder (Orion)']
elements: ['fire']
timeline_presence:
  original: false
  go: false
  ares: true
  orion: true
intended_use: [program_parsing, animation_modeling, lora_training, cross_character_comparison]
---
# Inamori Asuto

## 1. Overview
- **Role positioning:** Ares/Orion protagonist; sunshine-core mediator; energy and fairness driver.
- **Series scope:** Ares / Orion.
- **Narrative job:**
  - Re-centers the story on “soccer is fun / honest” when the environment turns system-driven or corrupted.
  - Connects fragmented teammates (including rivals with baggage).
  - Serves as the emotional counterbalance to colder strategists (e.g., Nosaka) and harsher rivals.

## 2. Identity & Canon Info
- **JP name:** 稲森 明日人
- **English/Dub name:** Sonny Wright
- **Gender:** Male
- **Age range (by timeline):**
  - Ares / Orion: Middle school (approx. 13–15)
- **Primary teams (timeline-split):**
  - Ares: Inakuni Raimon (temporary captain)
  - Orion: Inazuma Japan (Orion)
- **On-field position(s):**
  - Ares: Forward (FW)
  - Orion: Midfielder (MF) (role shifts toward linking and tempo)
- **Element:** Fire

## 3. Timeline Variants

### Original Timeline

```yaml
timeline_id: original
presence: no
```

- Not applicable: no canonical role for this character in this timeline variant within the provided sources.


### GO Timeline

```yaml
timeline_id: go
presence: no
```

- Not applicable: no canonical role for this character in this timeline variant within the provided sources.


### Ares Timeline

```yaml
timeline_id: ares
presence: yes
```

- **Age / state:**
  - Middle school forward; energy-forward leader.
  - Temporary captain framing for underdog team.
- **Identity shifts:**
  - From island-school underdog → national-level contender.
  - Identity is 'bright striker who lifts others'.
- **Personality emphasis:**
  - Upbeat, straightforward, friendly.
  - Pushes fairness even when outmatched.
- **Narrative function shifts:**
  - Underdog ignition: makes a weak team feel alive.
  - Turns match hardship into growth.


### Orion Timeline

```yaml
timeline_id: orion
presence: yes
```

- **Age / state:**
  - Middle school midfielder; more connective role.
  - Competes under sabotage/corruption pressure.
- **Identity shifts:**
  - Forward ace role shifts into midfield glue.
  - Becomes emotional stabilizer for a national team under stress.
- **Personality emphasis:**
  - Still bright, but more alert and protective.
  - Can become stubborn against injustice.
- **Narrative function shifts:**
  - Legitimacy beacon against match-fixing.
  - Connects star players into one functional unit.


## 4. Visual Design

### 4.1 Core visual traits
- **Hair shape:** Short dark gray hair with upward spikes on top/back.
- **Hair color:** Dark gray
- **Eyes:** Deep green
- **Build:** Lean, athletic; quick-footed silhouette.
- **Aura:** “Sun” energy—open smiles, direct eye contact, forward movement.

### 4.2 Outfits (by era)
- **Match kit (Ares):** Inakuni Raimon kit (underdog school identity).
- **Match kit (Orion):** Inazuma Japan kit; more formal “representative” staging.
- **Casual:** White T-shirt + short-sleeved blue hoodie + dark blue shorts (youthful, sporty).
- **Accessories:** Keep minimal; his identity reads through expression and hair/eyes.

## 5. Personality Model
- **Surface persona:** Cheerful, honest, easy to approach.
- **Core drives:** Protect friends; keep soccer meaningful; help others find their path.
- **Under stress:** Becomes stubbornly principled; refuses to bend to intimidation.
- **Interaction patterns:** Leads by encouraging participation; tends to pull quieter teammates into the light.

## 6. Narrative Function
- Protagonist “sun core”: stabilizes tone when the plot turns dark (Ares system logic, Orion corruption).
- A bridge character that makes redemption arcs possible for rivals.
- Ensures the story never becomes pure “cold strategy.”

## 7. Abilities & Play Style

### 7.1 Playstyle traits
- **Type:** All-round attacker; quick acceleration; high work rate.
- **Solo vs team:** Team-first; loves one-twos and link play, then finishes when needed.
- **Signature strengths:** Momentum creation, perseverance, opportunistic finishing.

### 7.2 Signature techniques (representative)
- **Dribble**
  - Inabikari Dash
- **Shoot**
  - Crazy Sunlight
  - Shining Penguin
  - Bakunetsu Storm
- **Team / Combo finish**
  - Twin Boost
  - Inazuma Break
- **Tactical / High-impact finisher**
  - Last Resort Σ
- **Defense / Counter**
  - Counterdrive
- **Utility**
  - Sunrise Blitz

## 8. Relationship Graph
- **Core relationships (1–3):**
  - Nosaka Yuuma — ideology contrast (heart vs strategy); learns mutual respect.
  - Haizaki Ryouhei — rival-ally tension; Asuto’s sincerity softens edges.
  - Coach/guide figures (e.g., Zhao Jinyun contexts) — shapes his team integration.
- **Opposition / foils:** System-first soccer logic; Orion manipulation tactics.
- **Growth influences:** Learns leadership under corruption pressure; shifts from “bright forward” to “team connector.”

## 9. Animation / AI Training Tags
```text
inazuma_inamori_asuto, anime_style, japanese_teen_boy,
forward_ares, midfielder_orion, fire_element,
short_darkgray_spiky_hair, deep_green_eyes,
sunny_smile, upbeat_energy, fairness_driven,
underdog_captain, team_connector,
timeline_ares, timeline_orion
```

## 10. Notes for Animation & LoRA Training
- **Emotion range:** bright optimism → righteous anger → exhausted but smiling perseverance.
- **Common actions:** open-hand invite, quick dash start, shoulder-to-shoulder encouragement, direct stare at cheaters.
- **Do-not-mix warnings:**
  - Keep Ares forward role vs Orion midfielder role separated if you’re training position-specific poses.
- **Easy-to-draw-wrong pitfalls:**
  - His identity relies on eyes + smile; if you make him too stoic, he stops reading as Asuto.
