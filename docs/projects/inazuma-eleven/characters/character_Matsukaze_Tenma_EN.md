---
schema_version: "inazuma_character_v1.0"
doc_type: "character_profile"
character_id: "matsukaze_tenma"
canonical_name: "Matsukaze Tenma"
jp_name: "松風 天馬"
english_name: "Arion Sherwind"
gender: "male"
primary_positions: ['midfielder', '(occasional) goalkeeper']
elements: ['wind']
timeline_presence:
  original: false
  go: true
  ares: false
  orion: false
intended_use: [program_parsing, animation_modeling, lora_training, cross_character_comparison]
---
# Matsukaze Tenma

## 1. Overview
- **Role positioning:** GO-era main protagonist; inspirational midfielder; “wind” motif leader.
- **Series scope:** GO (primary).
- **Narrative job:**
  - Rebuilds “true soccer” through empathy and stubborn idealism.
  - Functions as a **connector** who turns rivals into teammates.
  - Carries the GO power-system layers (Keshin / Armed / Mixi Max) as a showcase vessel.

## 2. Identity & Canon Info
- **JP name:** 松風 天馬
- **English/Dub name:** Arion Sherwind
- **Gender:** Male
- **Age range (by timeline):**
  - GO: Middle school (approx. 13–15)
  - Other timelines: Not applicable in provided sources
- **Primary teams (GO):**
  - Shinsei Raimon (captain)
  - Chrono Storm (captain)
  - Shinsei Inazuma Japan / Inazuma Japan (GO) (captain-facing role)
- **On-field position(s):** Midfielder (MF); occasional Goalkeeper in specific contexts
- **Element:** Wind

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
presence: yes
```

- **Age / state:**
  - Middle school midfielder; captain of the new Raimon.
  - Power-system carrier across arcs.
- **Identity shifts:**
  - From fanboy/outsider → symbolic leader of 'true soccer'.
  - Gains multi-form power states (Keshin, Armed, Mixi Max).
- **Personality emphasis:**
  - Friendly, caring, relentlessly hopeful.
  - Courage spikes when teammates are threatened.
- **Narrative function shifts:**
  - The ideological spearhead against system-control soccer.
  - Primary vehicle for GO mechanics and team unification.


### Ares Timeline

```yaml
timeline_id: ares
presence: no
```

- Not applicable: no canonical role for this character in this timeline variant within the provided sources.


### Orion Timeline

```yaml
timeline_id: orion
presence: no
```

- Not applicable: no canonical role for this character in this timeline variant within the provided sources.


## 4. Visual Design

### 4.1 Core visual traits
- **Hair shape:** Chestnut-brown hair shaped like wind swirls / wing-like curls; high-motion silhouette.
- **Hair color:** Chestnut brown
- **Eyes:** Large greyish-blue eyes
- **Build:** Slim athletic; springy movement quality.
- **Aura:** Open chest posture, forward-lean run; reads as “inviting” rather than intimidating.

### 4.2 Outfits (by era)
- **Match kit (GO):** Shinsei Raimon uniform; he is almost always shown in soccer kit (casual outfits are rare).
- **Child flashback outfit:** Red shirt with a puppy pattern + yellow sleeves; blue jeans (useful for flashback dataset).
- **Disguise outfits (Chrono Stone):** Multiple era-based disguises; treat each as separate sub-sets.
- **Mixi Max visual note:** Mixi Max forms change hair color/ornaments (e.g., Arthur Mixi Max adds blue hair clips and a different palette); separate training tags.

## 5. Personality Model
- **Surface persona:** Bright, kind, socially fearless; tries to help immediately.
- **Core drives:** Protect teammates; keep soccer honest; turn conflict into connection.
- **Under stress:** Becomes stubborn and brave; will throw himself into danger to shield others.
- **Interaction patterns:** Starts as a follower of “heroes,” grows into being one; persuades by sincerity rather than dominance.

## 6. Narrative Function
- GO’s “wind protagonist” template: momentum, change, freedom.
- A bridge between old ideals (Endou era) and new mechanics (Keshin/Mixi Max).
- Used to prove that empathy can beat authoritarian systems.

## 7. Abilities & Play Style

### 7.1 Playstyle traits
- **Type:** Technical + tactical midfielder; dribble-driven playmaker.
- **Solo vs team:** Highly team-oriented; prefers enabling others, then finishing when necessary.
- **Key strengths:** Rapid adaptation; stamina; clutch morale uplift.

### 7.2 Signature techniques (representative)
- **Dribble**
  - Soyokaze Step
  - Soyoyagi Step
  - Spiral Draw
- **Shoot**
  - Mach Wind
  - Hinawa Bullet
  - Evolution
- **Tactical / Team**
  - The Earth ∞
- **(Dark-contrast showcase)**
  - Black Dawn

## 8. Relationship Graph
- **Core relationships (1–3):**
  - Endou Mamoru (GO) — mentor/ideological origin point.
  - Shindou Takuto — captain/brain partner; stabilizes Tenma’s emotional drive.
  - Tsurugi Kyousuke — rival-ally; pushes Tenma’s grit and combativeness.
- **Opposition / foils:** Fifth Sector system-control ideology; characters who treat soccer as obedience.
- **Growth influences:** Learns to lead without copying heroes; becomes the hero.

## 9. Animation / AI Training Tags
```text
inazuma_matsukaze_tenma, anime_style, japanese_teen_boy,
midfielder, occasional_goalkeeper, wind_element,
chestnut_brown_hair, wind_swirl_hair, greyblue_eyes,
friendly_smile, energetic_run, empathy_leader,
mechanic_keshin, mechanic_keshin_armed, mechanic_miximax,
timeline_go
```

## 10. Notes for Animation & LoRA Training
- **Emotion range:** bright joy → serious determination → protective fear → relieved laughter.
- **Common actions:** open-arm invite gesture, quick small-step dribble, forward lean sprint, “shield teammate” body-block.
- **Do-not-mix warnings:**
  - Separate: base GO Tenma vs Mixi Max vs Armed forms (hair/ornament/palette changes).
  - Flashback child outfit is a distinct subset; don’t mix into match-kit LoRA unless you want outfit drift.
- **Easy-to-draw-wrong pitfalls:**
  - Hair must read as “wind/wing curls” (this is the character ID).
  - Keep his posture open; don’t turn him into a stoic/closed-off silhouette.
