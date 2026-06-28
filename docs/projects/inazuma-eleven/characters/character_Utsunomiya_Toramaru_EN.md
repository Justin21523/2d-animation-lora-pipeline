---
schema_version: "inazuma_character_v1.0"
doc_type: "character_profile"
character_id: "utsunomiya_toramaru"
canonical_name: "Utsunomiya Toramaru"
jp_name: "宇都宮 虎丸"
english_name: "Austin Hobbes"
gender: "male"
primary_positions: ['forward']
elements: ['forest']
timeline_presence:
  original: true
  go: true
  ares: false
  orion: false
intended_use: [program_parsing, animation_modeling, lora_training, cross_character_comparison]
---
# Utsunomiya Toramaru

## 1. Overview
- **Role positioning:** Young prodigy striker; “innocent potential” contrast; late-addition spark.
- **Series scope:** Original (main); GO (appearance as an older/associated figure).
- **Narrative job:**
  - Represents raw talent that needs guidance and confidence.
  - Adds a “kid among teens” emotional note: vulnerability + surprising explosiveness.
  - Often used to show the team’s ability to nurture and protect.

## 2. Identity & Canon Info
- **JP name:** 宇都宮 虎丸
- **English/Dub name:** Austin Hobbes
- **Gender:** Male
- **Age range (by timeline):**
  - Original: Young middle schooler / child-prodigy range (approx. 11–14)
  - GO: Older teen/young adult presentation (treat as separate dataset)
  - Ares / Orion: Not present in provided sources
- **Primary teams (timeline-split):**
  - Original: Raimon; Inazuma Japan
  - GO: Associated presence in GO-era contexts
- **On-field position:** Forward (FW)
- **Element:** Forest

## 3. Timeline Variants

### Original Timeline

```yaml
timeline_id: original
presence: yes
```

- **Age / state:**
  - Young prodigy forward; physically smaller but high potential.
  - Learns confidence through team support.
- **Identity shifts:**
  - From shy newcomer → reliable finisher in big matches.
  - Moves into national selection context.
- **Personality emphasis:**
  - Shy, sweet, eager to please.
  - Bravery grows as he feels protected.
- **Narrative function shifts:**
  - Innocence + talent: motivates older teammates to step up.
  - Provides surprise scoring threat.


### GO Timeline

```yaml
timeline_id: go
presence: yes
```

- **Age / state:**
  - GO-era appearance exists; presentation reads more formal/mature.
  - Often staged as connected to system/authority side in design.
- **Identity shifts:**
  - Identity shifts from 'kid striker' → 'older figure with formal styling'.
  - Keep this node separate from Original Toramaru.
- **Personality emphasis:**
  - Less naive externally; more composed.
  - Still gentle compared to typical authority characters.
- **Narrative function shifts:**
  - Functions as a bridge/cameo linking eras.
  - Adds continuity texture.


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
- **Hair shape:** Spiky blue-black hair; compact head silhouette.
- **Hair color:** Blue-black
- **Eyes:** Dark; often drawn large for “innocent” readability.
- **Build:** Smaller/younger proportions in Original; shorter limbs and lighter frame.
- **Aura:** Soft posture, slight inward shoulders when shy; switches to sharp forward lean when attacking.

### 4.2 Outfits (by era)
- **Match kit (Original):** Raimon / Inazuma Japan forward kit; emphasize “small striker” contrast.
- **Casual (Original):** White shirt + yellow jacket + grey shorts (youthful streetwear).
- **GO-era styling:** Grey formal outfit with black tie and black gloves (formal/authority-coded; separate training split).

## 5. Personality Model
- **Surface persona:** Shy, polite, innocent.
- **Core drives:** Wants to belong; wants approval from trusted older figures; loves soccer but doubts himself.
- **Under stress:** Freezes or hesitates unless reassured; once triggered, can burst into unexpectedly bold play.
- **Interaction patterns:** Attaches to mentors; responds strongly to praise and protection.

## 6. Narrative Function
- “Protected prodigy” slot: raises stakes by adding someone the team emotionally wants to keep safe.
- Demonstrates that teamwork can unlock talent, not just train it.
- Adds tonal sweetness to otherwise intense arcs.

## 7. Abilities & Play Style

### 7.1 Playstyle traits
- **Type:** Burst finisher with quick reactions; thrives on through-balls and second chances.
- **Solo vs team:** Team-dependent—best when teammates create space and feed him.

### 7.2 Signature techniques (representative)
- **Shoot**
  - Tiger Drive
  - Tiger Storm
  - Gladius Arch
- **Dribble / Link**
  - Hitori One-Two
- **Team / Combo finish**
  - Grand Fire
  - Jet Stream
- **Utility**
  - RC Shoot

## 8. Relationship Graph
- **Core relationships (1–3):**
  - Endou Mamoru — captain/guardian figure; provides safety and confidence.
  - Gouenji Shuuya — striker role model; sets the “ace” target Toramaru chases.
  - Utsunomiya Tae — family anchor (mother).
- **Opposition / foils:** Older, physically dominant defenders; intimidation pressure.
- **Growth influences:** Confidence gained through acceptance and repeated “it’s okay, try” reinforcement.

## 9. Animation / AI Training Tags
```text
inazuma_utsunomiya_toramaru, anime_style, japanese_boy,
forward, forest_element, young_prodigy,
spiky_blueblack_hair, innocent_expression, small_build,
tiger_motif, shy_posture,
timeline_original, timeline_go_formal
```

## 10. Notes for Animation & LoRA Training
- **Emotion range:** shy smile → startled fear → sudden brave push → proud relief.
- **Common actions:** small-step run-up, quick glance to captain, clutching fists near chest, sudden sprint burst.
- **Do-not-mix warnings:**
  - Original “kid proportions” vs GO “formal older styling” must be separated.
- **Easy-to-draw-wrong pitfalls:**
  - Keep him smaller than most teammates in Original scenes.
  - Hair should read blue-black (not pure black) to preserve the character ID.
