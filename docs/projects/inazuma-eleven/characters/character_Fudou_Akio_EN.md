---
schema_version: "inazuma_character_v1.0"
doc_type: "character_profile"
character_id: "fudou_akio"
canonical_name: "Fudou Akio"
jp_name: "不動 明王"
english_name: "Caleb Stonewall"
gender: "male"
primary_positions: ['midfielder']
elements: ['fire']
timeline_presence:
  original: true
  go: true
  ares: true
  orion: true
intended_use: [program_parsing, animation_modeling, lora_training, cross_character_comparison]
---
# Fudou Akio

## 1. Overview
- **Role positioning:** Tactical “joker” midfielder; provocation specialist; frequently the **dark mirror** of more idealistic leaders.
- **Series scope:** Original / GO / Ares / Orion (as separate continuity nodes).
- **Narrative job:**
  - Injects uncertainty: mind games, tempo manipulation, and rule-edge behavior.
  - Forces teammates to answer “Do we accept him?”—then becomes a high-value asset once integrated.
  - Provides a character template for “cynical strategist who still cares.”

## 2. Identity & Canon Info
- **JP name:** 不動 明王
- **English/Dub name:** Caleb Stonewall
- **Gender:** Male
- **Age range (by timeline):**
  - Original: Middle school (approx. 13–15)
  - GO: Adult/older teen context depending on appearance (treat as separate dataset)
  - Ares / Orion: Middle school (approx. 13–15) in alternate continuity
- **Primary teams (timeline-split):**
  - Original: Shin Teikoku Gakuen (captain); Inazuma Japan
  - GO: Appears in GO-era contexts (coach/support roles appear in some team lists)
  - Ares: Teikoku Gakuen / Teikoku-related reboot slot (eye color differs)
  - Orion: Appears under Orion continuity nodes
- **On-field position:** Midfielder (MF)
- **Element:** Fire

## 3. Timeline Variants

### Original Timeline

```yaml
timeline_id: original
presence: yes
```

- **Age / state:**
  - Middle school midfielder; antagonistic-to-ally arc structure.
  - Often positioned as a 'necessary evil' playmaker.
- **Identity shifts:**
  - From opponent-minded Teikoku captain → national-team contributor.
  - Builds value via tactical reads and dirty-work.
- **Personality emphasis:**
  - Sarcastic, confrontational, testing boundaries.
  - Shows loyalty once accepted.
- **Narrative function shifts:**
  - Conflict catalyst inside the team.
  - Provides tactical solutions others won’t attempt.


### GO Timeline

```yaml
timeline_id: go
presence: yes
```

- **Age / state:**
  - GO-era presence exists in provided sources; treat as separate age/role dataset.
  - Often framed in a more mature context.
- **Identity shifts:**
  - Identity leans toward veteran/mentor or system-side experience.
  - Less about 'new kid villain', more about 'experienced operator'.
- **Personality emphasis:**
  - Still cynical, but more controlled.
  - Manipulation becomes strategic rather than petty.
- **Narrative function shifts:**
  - Adds tactical weight and worldliness to GO-era casts.
  - A foil to pure idealists.


### Ares Timeline

```yaml
timeline_id: ares
presence: yes
```

- **Age / state:**
  - Middle school reboot node; visual differences are explicit.
  - Eye color differs compared to Original.
- **Identity shifts:**
  - Alternate continuity version; do not merge with Original Fudou.
  - Team slot can shift while keeping the same archetype.
- **Personality emphasis:**
  - Sharper edge; competitive ruthlessness emphasized.
  - Still shows hints of caring through actions, not words.
- **Narrative function shifts:**
  - Reboots the 'joker strategist' function for Ares pacing.
  - Used to raise the tactical IQ of matches.


### Orion Timeline

```yaml
timeline_id: orion
presence: yes
```

- **Age / state:**
  - Orion continuity node; high-stakes international environment.
  - Tactics exist alongside corruption/sabotage framing.
- **Identity shifts:**
  - Strategist role becomes anti-cheat / counter-manipulation tool.
  - Operates as someone who understands dirty play.
- **Personality emphasis:**
  - Cynical realism; less shocked by sabotage.
  - Can be protective when teammates are targeted.
- **Narrative function shifts:**
  - Used as an internal counter to Orion-style manipulation.
  - Turns 'playing dirty' knowledge into defense for the team.


## 4. Visual Design

### 4.1 Core visual traits
- **Hair shape:** Brown mohawk / sharp upward ridge (aggressive silhouette).
- **Hair color:** Brown
- **Eyes:** Blue-gray in Original; explicitly different in Ares (dark green).
- **Build:** Average height, lean-athletic; reads wiry rather than bulky.
- **Aura:** Side-eye smirk; posture slightly angled like he’s always planning something.

### 4.2 Outfits (by era)
- **Match kit:** Midfielder kit; often staged centrally with “puppetmaster” framing.
- **Casual:** Darker streetwear; the silhouette should read “edgy strategist”.
- **Ares visual warning:** Eye color change is a hard variant split (do not average them in training).
- **GO/Orion:** Treat as separate wardrobe language (more mature/official contexts can appear).

## 5. Personality Model
- **Surface persona:** Cynical, provocative, sarcastic; enjoys pushing buttons.
- **Core drives:** Control through information; avoid vulnerability; prove intelligence matters more than “spirit”.
- **Under stress:** Becomes sharper and more ruthless; may isolate rather than seek comfort.
- **Interaction patterns:** Tests people before trusting them; loyalty is expressed as action (covering a teammate, making a risky pass), not words.

## 6. Narrative Function
- Team destabilizer turned tactical asset.
- A consistent foil to “pure-hearted captain” archetypes: he forces moral and strategic negotiation.
- A reusable template for “antagonist energy inside the protagonist team.”

## 7. Abilities & Play Style

### 7.1 Playstyle traits
- **Type:** Tactical playmaker / disruptor.
- **Strengths:** Tempo control, psychological warfare, opportunistic positioning, high-risk high-reward decisions.
- **Team orientation:** Works best when the team can absorb his edge without collapsing.

### 7.2 Signature techniques (representative)
- **Tactical / Support**
  - Ike Ike!
  - Triple Boost
  - Judge Through 2
- **Defense**
  - Killer Slide
- **Shoot / Finisher**
  - Koutei Penguin 2gou
  - Koutei Penguin 2gou feat. Shark
- **Tactical (zone / pressure)**
  - Death Crusher Zone

## 8. Relationship Graph
- **Core relationships (1–3):**
  - Kidou Yuuto — brain-vs-brain foil; mutual tactical respect with friction.
  - Endou Mamoru — moral counterweight who forces Fudou to “belong” rather than exploit.
  - Teikoku/Teikoku-associated peers — identity anchor for his competitive pride.
- **Opposition / foils:** Idealists who refuse dirty play; authority systems he tries to outsmart.
- **Growth influences:** Learns to redirect cynical intelligence into team benefit.

## 9. Animation / AI Training Tags
```text
inazuma_fudou_akio, anime_style, japanese_teen_boy,
midfielder, tactical_playmaker, provocateur,
brown_mohawk, sharp_silhouette, sarcastic_smirk,
bluegray_eyes_original, darkgreen_eyes_ares,
fire_element, mind_games,
timeline_original, timeline_go, timeline_ares, timeline_orion
```

## 10. Notes for Animation & LoRA Training
- **Emotion range:** smug amusement → cold calculation → sudden flare of anger → reluctant pride.
- **Common actions:** sideways grin, finger-point taunt, sudden stop-and-turn passes, “fake compliance then betray expectation” movement.
- **Do-not-mix warnings:**
  - Eye color difference between Original vs Ares must be kept clean.
  - If using GO-era mature look, keep it separate from middle-school proportions and uniforms.
- **Easy-to-draw-wrong pitfalls:**
  - Mohawk ridge shape: don’t soften it into generic spiky hair.
  - Expressions: his face reads best with minimal exaggeration—subtle smirk > cartoon grin.
