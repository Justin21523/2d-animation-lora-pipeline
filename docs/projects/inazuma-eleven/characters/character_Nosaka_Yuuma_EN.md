---
schema_version: "inazuma_character_v1.0"
doc_type: "character_profile"
character_id: "nosaka_yuuma"
canonical_name: "Nosaka Yuuma"
jp_name: "野坂 悠馬"
english_name: "Heath Moore"
gender: "male"
primary_positions: ['midfielder', 'forward (secondary)']
elements: ['fire', 'mountain']
timeline_presence:
  original: false
  go: false
  ares: true
  orion: true
intended_use: [program_parsing, animation_modeling, lora_training, cross_character_comparison]
---
# Nosaka Yuuma

## 1. Overview
- **Role positioning:** Tactical core / “emperor strategist”; calm commander with a ruthless edge when necessary.
- **Series scope:** Ares / Orion.
- **Narrative job:**
  - Provides the “brain of the team” slot: match-reading, exploitation of patterns, rapid re-planning.
  - Serves as ideological contrast: results-logic vs the protagonists’ emotional/fair-play axis.
  - Adds tension through secrecy and high-pressure decision-making (including health-related constraints in canon descriptions).

## 2. Identity & Canon Info
- **JP name:** 野坂 悠馬
- **English/Dub name:** Heath Moore
- **Gender:** Male
- **Age range (by timeline):**
  - Ares / Orion: Middle school (approx. 13–15)
- **Primary teams (timeline-split):**
  - Ares: Outei Tsukinomiya (captain)
  - Orion: Inazuma Japan (Orion) (secondary captain), Zhao Jinyuns (club context)
- **On-field position(s):** Midfielder (primary); Forward (secondary)
- **Element(s):** Fire; Mountain

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
  - Middle school strategist; captain of an elite program team.
  - Controls tempo and win conditions.
- **Identity shifts:**
  - Leader identity is 'tactics first'.
  - Relies on systematic planning and opponent exploitation.
- **Personality emphasis:**
  - Confident, cool, sometimes sarcastic.
  - Willing to play ruthlessly if it guarantees victory.
- **Narrative function shifts:**
  - Raises the tactical ceiling of the arc.
  - Acts as a foil to 'win with spirit' ideology.


### Orion Timeline

```yaml
timeline_id: orion
presence: yes
```

- **Age / state:**
  - Middle school national-team strategic pillar.
  - Plays inside a corruption/sabotage environment.
- **Identity shifts:**
  - Captain-adjacent role becomes 'keep the team coherent under dirty tactics'.
  - Adapts strategy to counter manipulation.
- **Personality emphasis:**
  - More openly cooperative over time.
  - Still sharp-tongued; uses psychological pressure when needed.
- **Narrative function shifts:**
  - Strategic shield: prevents chaos from breaking the team.
  - Turns Orion’s tricks into predictable patterns to exploit.


## 4. Visual Design

### 4.1 Core visual traits
- **Hair shape:** Short blond hair swept left; neat, controlled shape.
- **Hair color:** Blonde
- **Eyes:** Lightless/grey-toned eyes (cool, analytical stare).
- **Build:** Tall, slender; “elite student-athlete” proportions.
- **Aura:** Minimal expression; posture upright; reads like a chess player in human form.
- **Accessories:** Pendant necklace + black bracelet (keep consistent); shoes noted as yellow with white stripe in some Orion visuals.

### 4.2 Outfits (by era)
- **Match kit (Ares):** Outei Tsukinomiya kit; elite-team polish.
- **Match kit (Orion):** Inazuma Japan kit; often framed as the field commander.
- **Casual:** Clean, minimal fashion; avoid clutter—his design reads through silhouette + eyes.
- **“Commander” staging:** Often center-frame, hands in pockets or arms folded; treat as part of his visual language.

## 5. Personality Model
- **Surface persona:** Calm, confident, detached; can be sarcastic.
- **Core drives:** Control outcomes; avoid helplessness; prove intellect can dominate chaos.
- **Under stress:** Becomes ruthlessly efficient; may sacrifice niceties and relationships for the plan.
- **Interaction patterns:** Leads through instructions and predictions; respects people who can keep up mentally.

## 6. Narrative Function
- The franchise’s “tactical emperor” template: strategy as spectacle.
- A foil to heart-driven protagonists; forces them to mature tactically.
- In Orion, becomes the team’s anti-corruption counterplanner.

## 7. Abilities & Play Style

### 7.1 Playstyle traits
- **Type:** Tactical playmaker; tempo controller; precision passer with opportunistic finishing.
- **Solo vs team:** Team-oriented but command-driven—he orchestrates others like pieces.
- **Key strength:** Converts chaos into a solvable structure mid-match.

### 7.2 Signature techniques (representative)
- **Shoot**
  - King's Lance
  - Last Resort Σ
- **Dribble / Movement**
  - Presto Turn
  - Sky Walk
- **Defense / Counter**
  - Gekkoumaru Tsubame Gaeshi
- **Shoot (elemental / special)**
  - Blizzartornado
- **Team / Combo finish**
  - Inazuma Break

## 8. Relationship Graph
- **Core relationships (1–3):**
  - Inamori Asuto — heart/values contrast; forces Nosaka to re-evaluate what “winning” means.
  - Haizaki Ryouhei — high-friction ally/rival; both are sharp, but with different priorities.
  - Teammates who can execute complex plans — Nosaka’s trust is earned by competence.
- **Opposition / foils:** Cheating systems (Orion), emotional leaders who reject calculation.
- **Growth influences:** Moves from isolated strategist to cooperative commander.

## 9. Animation / AI Training Tags
```text
inazuma_nosaka_yuuma, anime_style, japanese_teen_boy,
midfielder, tactical_emperor, field_commander,
tall_slender_build, swept_left_blonde_hair, grey_dry_eyes,
cool_expression, sarcastic_tone,
pendant_necklace, black_bracelet,
fire_element, mountain_element,
timeline_ares, timeline_orion
```

## 10. Notes for Animation & LoRA Training
- **Emotion range:** cold calculation → sharp irritation → rare soft smile → intense “win now” glare.
- **Common actions:** finger-to-chin thinking pose, quick head turn to read formation, short precise pointing, minimal step adjustments.
- **Do-not-mix warnings:**
  - Keep “elite captain (Ares)” vs “national strategist (Orion)” separate if you rely on uniform cues and role poses.
- **Easy-to-draw-wrong pitfalls:**
  - Don’t over-animate his face; he reads strongest with subtle changes.
  - Hair sweep direction is identity-critical (left-swept silhouette).
