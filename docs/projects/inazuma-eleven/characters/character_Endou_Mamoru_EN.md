---
schema_version: "inazuma_character_v1.0"
doc_type: "character_profile"
character_id: "endou_mamoru"
canonical_name: "Endou Mamoru"
jp_name: "円堂 守"
english_name: "Mark Evans"
gender: "male"
primary_positions: ['goalkeeper', 'libero']
elements: ['mountain', 'wind (SD-event-only)']
timeline_presence:
  original: true
  go: true
  ares: true
  orion: true
intended_use: [program_parsing, animation_modeling, lora_training, cross_character_comparison]
---
# Endou Mamoru

## 1. Overview
- **Role positioning:** Main protagonist archetype (Original), team **captain-heart** and defensive anchor; later becomes a mentor/coach figure.
- **Series scope:** Original / GO / Ares / Orion.
- **Narrative job (what the story uses him for):**
  - The “never-give-up” **emotional stabilizer** who keeps the team moving when morale collapses.
  - A **goalkeeping wall** that turns impossible shots into “we’re still alive” moments.
  - A values benchmark: “soccer should be fun / played earnestly” vs systems that try to control it.

## 2. Identity & Canon Info
- **JP name:** 円堂 守
- **English/Dub name:** Mark Evans
- **Gender:** Male
- **Age range (by timeline):**
  - Original: Middle school (approx. 13–15)
  - GO: Adult (approx. early–mid 20s)
  - Ares / Orion: Middle school (approx. 13–15)
- **Primary teams (timeline-split):**
  - Original: Raimon (captain), Inazuma Japan (captain)
  - GO: Shinsei Raimon (coach; temporary GK), Team Raimon / Chrono Storm (coach roles in spin-off contexts)
  - Ares: Raimon (alternate continuity; captain role preserved)
  - Orion: Inazuma Japan (captain)
- **On-field position(s):** Goalkeeper; occasional Libero
- **Element:** Mountain (primary); Wind appears as a limited/event flag in some game contexts

## 3. Timeline Variants

### Original Timeline

```yaml
timeline_id: original
presence: yes
```

- **Age / state:**
  - Middle school goalkeeper; core captain role.
  - Identity is 'team heart + last line of defense'.
- **Identity shifts:**
  - Captain of Raimon; becomes the face of the team’s legitimacy.
  - Transitions into national representative captain (FFI arc).
- **Personality emphasis:**
  - Open, friendly, relentlessly optimistic.
  - Confidence grows from 'trying hard' into 'I can carry pressure for everyone'.
- **Narrative function shifts:**
  - Moves the plot forward by refusing to quit.
  - Turns defeats into training goals for the whole cast.


### GO Timeline

```yaml
timeline_id: go
presence: yes
```

- **Age / state:**
  - Adult / retired-from-frontline vibe; mentor authority.
  - Often positioned as coach/guide rather than field solution.
- **Identity shifts:**
  - From player-captain → coach/mentor symbol.
  - Represents continuity of 'true soccer' values.
- **Personality emphasis:**
  - More measured; less hot-blooded, more reassuring.
  - Still stubborn about fairness and effort.
- **Narrative function shifts:**
  - Functions as ideological anchor for the new generation.
  - Pushes the protagonists to reclaim soccer from systems.


### Ares Timeline

```yaml
timeline_id: ares
presence: yes
```

- **Age / state:**
  - Middle school goalkeeper in alternate continuity.
  - Captain identity remains a central constant.
- **Identity shifts:**
  - Same name/role slot but in a rebooted context.
  - Leads Raimon under different competitive conditions.
- **Personality emphasis:**
  - Upbeat leadership; quick trust-building.
  - More 'coachable' tone around structured tactics compared to Original.
- **Narrative function shifts:**
  - Acts as the franchise’s familiar heart in the reboot.
  - Keeps the story grounded in classic Inazuma values.


### Orion Timeline

```yaml
timeline_id: orion
presence: yes
```

- **Age / state:**
  - Middle school national-team captain presence.
  - Operates under higher political/organizational pressure.
- **Identity shifts:**
  - Captain role extended to international stage (FFI context).
  - Must manage team cohesion under sabotage/corruption pressure.
- **Personality emphasis:**
  - Calm under chaos; protective of teammates.
  - Less naive—more alert to manipulation.
- **Narrative function shifts:**
  - Captain-as-shield: absorbs pressure so the team can play.
  - Embodies fair play against match-fixing narratives.


## 4. Visual Design

### 4.1 Core visual traits
- **Hair shape:** Short brown hair with strong side spikes; a single bang hanging over the forehead.
- **Hair color:** Brown
- **Eyes:** Large, round brown eyes
- **Build:** Compact/athletic “everyday kid” proportions (not lanky, not bulky).
- **Silhouette / aura:** Forward-leaning, ready stance; reads as energetic and inviting rather than intimidating.
- **Iconic accessory:** Orange headband (treated as a trademark).

### 4.2 Outfits (by era)
- **Match kit (Original):**
  - Raimon goalkeeper kit; gloves always present; headband often kept even in matches.
  - Inazuma Japan kit in international context (captain presence: armband/leadership framing).
- **Casual (Original):** Sporty, practical streetwear; headband frequently retained.
- **GO adult / coach look:** Slightly longer spikier hair; casual jacket + jeans; collar-up habit reads as “cool mentor”.
- **Ares/Orion:** Reboot-era kits; keep headband + “friendly captain” silhouette consistent.

## 5. Personality Model
- **Surface persona (how he reads):**
  - Cheerful, approachable, “let’s do this together” energy.
- **Core drives (what pushes decisions):**
  - Protect the team; keep soccer meaningful; prove effort can beat despair.
- **Under stress:**
  - Doubles down on responsibility; turns panic into short, actionable goals (“one save at a time”).
- **Interaction patterns:**
  - Builds trust fast; treats rivals as future friends; motivates by praising effort, not just results.

## 6. Narrative Function
- The franchise’s **heart-and-hands** template: emotional leadership + literal goalkeeping.
- A “moral center” used to contrast against: cheating, system control, cynicism, or pure results-logic.
- A stabilizing node that makes large casts feel like one team.

## 7. Abilities & Play Style

### 7.1 Playstyle traits
- **Type:** Defensive specialist (shot-stopper) with leadership pressure tolerance.
- **Decision style:** Reads situations quickly; prioritizes team confidence and momentum.
- **Solo vs team:** Team-oriented—his saves are framed as “team survives”, not “I’m amazing”.

### 7.2 Signature techniques (representative)
- **Defense / Catch**
  - God Hand
  - Majin The Hand
  - Seigi no Tekken
  - Ijigen The Hand
- **Defense / Block / Team defense**
  - Triple Defense
  - Megaton Head
- **Tactical / Team**
  - The Earth
  - Jet Stream (team-oriented finisher context)
- **(Occasional) Shoot / Utility**
  - Inazuma 1gou

## 8. Relationship Graph
- **Core relationships (1–3):**
  - Gouenji Shuuya — key teammate/ace; Endou’s trust unlocks the “team becomes real” switch.
  - Kidou Yuuto — strategy counterpart; Endou is heart, Kidou is brain.
  - Endou Daisuke — legacy/mentor symbol shaping Endou’s ideals.
- **Opposition / foils:**
  - System-control antagonists (GO-era governance, Orion-era corruption) that challenge “fair soccer”.
- **Growth influences:**
  - Learns to combine optimism with tactical realism; becomes a mentor figure later.

## 9. Animation / AI Training Tags
```text
inazuma_endou_mamoru, anime_style, japanese_teen_boy,
goalkeeper, libero_variant, captain, orange_headband,
spiky_brown_hair, big_round_brown_eyes, mountain_element,
determined_smile, morale_leader,
timeline_original, timeline_go_adult, timeline_ares, timeline_orion
```

## 10. Notes for Animation & LoRA Training
- **Emotion range to capture:** bright optimism → fierce focus → protective anger → relieved laughter.
- **Common actions:** glove-adjust, forward lean before a save, wide stance, clenched fist “we can do it” signal.
- **Do-not-mix warnings:**
  - GO adult/coach version should be isolated from middle-school versions (face maturity + outfit + aura).
  - If training “captain-headband” identity, keep headband visibility consistent; it’s a recognition anchor.
- **Easy-to-draw-wrong pitfalls:**
  - Side spikes + single bang shape (silhouette is the ID).
  - Headband placement and thickness (don’t turn it into a thin hairpin line).
