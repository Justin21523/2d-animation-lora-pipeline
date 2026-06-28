---
schema_version: "inazuma_character_v1.0"
doc_type: "character_profile"
character_id: "gouenji_shuuya"
canonical_name: "Gouenji Shuuya"
jp_name: "豪炎寺 修也"
english_name: "Axel Blaze"
gender: "male"
primary_positions: ['forward']
elements: ['fire']
timeline_presence:
  original: true
  go: true
  ares: true
  orion: true
intended_use: [program_parsing, animation_modeling, lora_training, cross_character_comparison]
---
# Gouenji Shuuya

## 1. Overview
- **Role positioning:** Ace striker icon; “cool competence” contrast to louder teammates; later becomes a **masked/system figure** in GO.
- **Series scope:** Original / GO / Ares / Orion.
- **Narrative job:**
  - Provides the franchise’s signature “ace shot” spectacle and late-match swing potential.
  - Acts as a maturity/discipline benchmark for younger players.
  - In GO, functions as a **power-withheld authority node** (Holy Emperor identity) that tests the protagonist’s values.

## 2. Identity & Canon Info
- **JP name:** 豪炎寺 修也  
  - GO masked identity: イシド シュウジ (Ishido Shuuji)
- **English/Dub name:** Axel Blaze  
  - GO masked identity: Alex Zabel
- **Gender:** Male
- **Age range (by timeline):**
  - Original: Middle school (approx. 13–15)
  - GO: Adult (approx. early–mid 20s) under an official/authority presentation
  - Ares / Orion: Middle school (approx. 13–15) in alternate continuity
- **Primary teams (timeline-split):**
  - Original: Raimon; Inazuma Japan
  - GO: Fifth Sector leadership-facing identity (Holy Emperor / “Ishido” role)
  - Ares: Kidokawa Seishuu (Ares)
  - Orion: Inazuma Japan (Orion continuity)
- **On-field position:** Forward (FW)
- **Element:** Fire

## 3. Timeline Variants

### Original Timeline

```yaml
timeline_id: original
presence: yes
```

- **Age / state:**
  - Middle school forward; ace finisher.
  - Often framed as the team’s reliable 'closer'.
- **Identity shifts:**
  - From reluctant participant → committed ace.
  - Personal stakes fuel his discipline and intensity.
- **Personality emphasis:**
  - Cool, calm, precise.
  - Protective streak hidden under a controlled exterior.
- **Narrative function shifts:**
  - The ace shot that makes underdogs believable.
  - A role-model foil to less disciplined players.


### GO Timeline

```yaml
timeline_id: go
presence: yes
```

- **Age / state:**
  - Adult; presented as a high-authority figure within soccer governance.
  - Identity includes 'Ishido Shuuji' masking.
- **Identity shifts:**
  - Gouenji persona becomes separated from public identity.
  - Operates inside the system he appears to represent.
- **Personality emphasis:**
  - Colder and more distant externally.
  - Internally still value-driven; strategic withholding.
- **Narrative function shifts:**
  - A narrative 'gate' character: forces protagonists to confront system control.
  - Reframes the ace archetype as authority/structure.


### Ares Timeline

```yaml
timeline_id: ares
presence: yes
```

- **Age / state:**
  - Middle school ace forward in rebooted context.
  - Team slot differs (Kidokawa Seishuu).
- **Identity shifts:**
  - Same ace-striker function but with different competitive environment.
  - More openly integrated into team systems earlier.
- **Personality emphasis:**
  - Cool-headed, self-disciplined.
  - Less mystery; more 'elite athlete' vibe.
- **Narrative function shifts:**
  - Acts as a known franchise icon transplanted into a new bracket.
  - Raises baseline skill ceiling for the cast.


### Orion Timeline

```yaml
timeline_id: orion
presence: yes
```

- **Age / state:**
  - Middle school forward in international sabotage-heavy context.
  - Elite finisher under political pressure.
- **Identity shifts:**
  - Must operate in matches with manipulation/dirty tactics surrounding.
  - Ace role becomes 'win despite corruption'.
- **Personality emphasis:**
  - Controlled aggression; tactical patience.
  - Protective toward teammates under threat.
- **Narrative function shifts:**
  - Spectacle + legitimacy: his presence signals “this team is serious”.
  - A foil to Orion-style cheating and coercion.


## 4. Visual Design

### 4.1 Core visual traits
- **Hair shape:** Spiky light-blonde hair with flame-like silhouette; very high recognizability.
- **Hair color:** Light blonde
- **Eyes:** Dark brown
- **Build:** Athletic, slightly taller/leaner than average teen; “sprinter striker” vibe.
- **Aura:** Reserved confidence; minimal facial movement; intensity shows through eyes and posture.
- **Signature habit:** Often raises/adjusts the collar of soccer shirts (gives a sharp silhouette).

### 4.2 Outfits (by era)
- **Match kit (Original):**
  - Raimon forward kit; collar-up habit is a key animation tick.
  - Inazuma Japan kit; often framed as ace (center-forward staging).
- **Casual (Original):**
  - Orange jacket layered over a hoodie; brown pants; red sneakers (sporty streetwear).
- **GO authority identity (Ishido Shuuji):**
  - Formal/official styling; treat as a different character layer in training (masked identity vibe).
- **Ares/Orion:**
  - Kidokawa Seishuu (Ares) kit; Orion-era national kit as ace forward.

## 5. Personality Model
- **Surface persona:** Quiet, cool, competent; speaks less, acts more.
- **Core drives:** Protect loved ones; prove himself through disciplined play; keep promises.
- **Under stress:** Becomes hyper-focused; emotional displays are rare but sharp when triggered.
- **Interaction patterns:** Encourages by example; softens mainly around trusted teammates or family-related topics.

## 6. Narrative Function
- “Ace striker icon” template for the franchise: the shot everyone recognizes.
- Contrast engine: cool professionalism vs emotional/chaotic teammates.
- In GO: transforms into a **system/authority mirror**—forces the protagonists to define what “real soccer” means.

## 7. Abilities & Play Style

### 7.1 Playstyle traits
- **Type:** Explosive finisher with strong shot power and clean mechanics.
- **Orientation:** Mostly solo-finisher, but supports team combo shots when trust is established.
- **Key strength:** Reliable execution in high-pressure finales.

### 7.2 Signature techniques (representative)
- **Shoot**
  - Fire Tornado
  - Bakunetsu Screw
  - Maximum Fire
  - Final Tornado
- **Defense**
  - Heat Tackle
- **Team / Combo finish**
  - Twin Boost
  - Jet Stream
  - Inazuma Break
  - Crossfire

## 8. Relationship Graph
- **Core relationships (1–3):**
  - Endou Mamoru — captain who earns his trust; emotional counterbalance.
  - Gouenji Yuuka — sister; major motivator for his choices and discipline.
  - Kidou Yuuto — tactical ally; enables higher-level team play.
- **Opposition / foils:**
  - Characters/systems that treat soccer as control, not passion (especially GO governance layer).
- **Growth influences:**
  - Learns to rejoin and rely on a team rather than carrying everything alone.

## 9. Animation / AI Training Tags
```text
inazuma_gouenji_shuuya, anime_style, japanese_teen_boy,
forward, ace_striker, fire_element,
spiky_blonde_hair, flame_hair_silhouette, dark_brown_eyes,
cool_expression, collar_up_habit,
timeline_original, timeline_go_masked_identity, timeline_ares, timeline_orion
```

## 10. Notes for Animation & LoRA Training
- **Emotion range:** calm → cold focus → restrained anger → rare gentle smile.
- **Common actions:** collar flick/raise, low-head gaze before shooting, compact sprint start, clean follow-through.
- **Do-not-mix warnings:**
  - GO masked identity (Ishido) should be separated from teen Gouenji; different aura, role, and wardrobe language.
- **Easy-to-draw-wrong pitfalls:**
  - Hair silhouette must read “flames” even in motion.
  - Keep the calm face—don’t over-exaggerate expressions unless it’s a deliberate break moment.
