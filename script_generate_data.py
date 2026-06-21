import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 18000

# === 100% ÉQUILIBRÉ ===
sizes = [3000, 3000, 3000, 3000, 3000, 3000]   # Exactement 3000 par profil
labels = ['Absolute_Beginner', 'Beginner', 'Low_Intermediate', 
          'Intermediate', 'Advanced', 'Expert']

def generate_player_data(n, level):
    if level == 'Absolute_Beginner':
        hints_used         = np.random.normal(19, 6, n).clip(0, 45).astype(int)
        correct_answers    = np.random.normal(11, 6, n).clip(2, 30).astype(int)
        wrong_answers      = np.random.normal(19, 7, n).clip(6, 50).astype(int)
        objectives_completed = np.random.normal(5, 3, n).clip(1, 15).astype(int)
        knowledge_score    = np.random.normal(30, 11, n).clip(5, 50)
        progression_rate   = np.random.normal(0.25, 0.12, n).clip(0.05, 0.50)
        retry_after_fail   = np.random.normal(13, 6, n).clip(3, 35).astype(int)
        load_time          = np.random.normal(3.8, 1.7, n).clip(1.0, 13)
        crash_count        = np.random.normal(4, 3, n).clip(0, 22).astype(int)
        lag_events         = np.random.normal(13, 8, n).clip(1, 45).astype(int)
        frame_drops        = np.random.normal(19, 12, n).clip(2, 65).astype(int)
        api_errors         = np.random.normal(5.5, 4, n).clip(0, 28).astype(int)
        
        playtime_voluntary = np.random.normal(32, 18, n).clip(5, 85).astype(int)
        bonus_collected    = np.random.normal(7, 5, n).clip(0, 25).astype(int)
        challenges_attempted = np.random.normal(9, 5, n).clip(1, 25).astype(int)
        idle_time          = np.random.normal(260, 120, n).clip(80, 650)
        exploration_rate   = np.random.normal(0.52, 0.15, n).clip(0.15, 0.80)
        combo_count        = np.random.normal(2.5, 2.5, n).clip(0, 12).astype(int)
        skip_count         = np.random.normal(15, 7, n).clip(4, 45).astype(int)
        help_requests      = np.random.normal(19, 8, n).clip(6, 50).astype(int)
        give_up_count      = np.random.normal(10, 5, n).clip(3, 28).astype(int)
        pause_count        = np.random.normal(27, 13, n).clip(10, 75).astype(int)
        total_pause_time   = np.random.normal(620, 290, n).clip(120, 1900)
        focus_time         = np.random.normal(1100, 650, n).clip(150, 3200)
        frustration_events = np.random.normal(21, 9, n).clip(6, 55).astype(int)
        session_count      = np.random.normal(5, 3, n).clip(1, 18).astype(int)
        days_active        = np.random.normal(8, 5, n).clip(1, 30).astype(int)

    elif level == 'Beginner':
        hints_used         = np.random.normal(12, 5, n).clip(0, 30).astype(int)
        correct_answers    = np.random.normal(18, 7, n).clip(5, 42).astype(int)
        wrong_answers      = np.random.normal(14, 6, n).clip(4, 38).astype(int)
        objectives_completed = np.random.normal(8, 4, n).clip(2, 20).astype(int)
        knowledge_score    = np.random.normal(43, 12, n).clip(15, 65)
        progression_rate   = np.random.normal(0.39, 0.14, n).clip(0.12, 0.65)
        retry_after_fail   = np.random.normal(8, 4, n).clip(0, 22).astype(int)
        load_time          = np.random.normal(2.7, 1.2, n).clip(0.6, 8)
        crash_count        = np.random.normal(2.1, 2, n).clip(0, 13).astype(int)
        lag_events         = np.random.normal(8, 5, n).clip(0, 26).astype(int)
        frame_drops        = np.random.normal(12, 8, n).clip(0, 42).astype(int)
        api_errors         = np.random.normal(3, 3, n).clip(0, 16).astype(int)
        
        playtime_voluntary = np.random.normal(50, 25, n).clip(12, 125).astype(int)
        bonus_collected    = np.random.normal(13, 8, n).clip(1, 42).astype(int)
        challenges_attempted = np.random.normal(15, 7, n).clip(4, 38).astype(int)
        idle_time          = np.random.normal(175, 80, n).clip(35, 460)
        exploration_rate   = np.random.normal(0.64, 0.13, n).clip(0.3, 0.88)
        combo_count        = np.random.normal(6, 4, n).clip(0, 22).astype(int)
        skip_count         = np.random.normal(9, 5, n).clip(1, 26).astype(int)
        help_requests      = np.random.normal(12, 6, n).clip(3, 32).astype(int)
        give_up_count      = np.random.normal(6, 4, n).clip(0, 19).astype(int)
        pause_count        = np.random.normal(18, 10, n).clip(4, 52).astype(int)
        total_pause_time   = np.random.normal(410, 210, n).clip(70, 1250)
        focus_time         = np.random.normal(1850, 850, n).clip(350, 4600)
        frustration_events = np.random.normal(14, 7, n).clip(4, 38).astype(int)
        session_count      = np.random.normal(9, 5, n).clip(2, 26).astype(int)
        days_active        = np.random.normal(15, 8, n).clip(4, 48).astype(int)

    elif level == 'Low_Intermediate':
        hints_used         = np.random.normal(7, 3.5, n).clip(0, 20).astype(int)
        correct_answers    = np.random.normal(29, 9, n).clip(15, 55).astype(int)
        wrong_answers      = np.random.normal(9, 4.5, n).clip(2, 24).astype(int)
        objectives_completed = np.random.normal(12, 5, n).clip(5, 26).astype(int)
        knowledge_score    = np.random.normal(59, 10, n).clip(42, 78)
        progression_rate   = np.random.normal(0.60, 0.13, n).clip(0.38, 0.82)
        retry_after_fail   = np.random.normal(5, 3, n).clip(0, 14).astype(int)
        load_time          = np.random.normal(2.0, 0.9, n).clip(0.5, 5.5)
        crash_count        = np.random.normal(1.1, 1.2, n).clip(0, 7).astype(int)
        lag_events         = np.random.normal(5.5, 4, n).clip(0, 18).astype(int)
        frame_drops        = np.random.normal(7.5, 5.5, n).clip(0, 28).astype(int)
        api_errors         = np.random.normal(1.7, 1.8, n).clip(0, 11).astype(int)
        
        playtime_voluntary = np.random.normal(78, 30, n).clip(35, 185).astype(int)
        bonus_collected    = np.random.normal(21, 10, n).clip(6, 52).astype(int)
        challenges_attempted = np.random.normal(21, 8, n).clip(9, 46).astype(int)
        idle_time          = np.random.normal(125, 60, n).clip(25, 340)
        exploration_rate   = np.random.normal(0.75, 0.10, n).clip(0.48, 0.93)
        combo_count        = np.random.normal(16, 8, n).clip(5, 42).astype(int)
        skip_count         = np.random.normal(5.5, 3.5, n).clip(0, 16).astype(int)
        help_requests      = np.random.normal(7.5, 4, n).clip(1, 20).astype(int)
        give_up_count      = np.random.normal(3.2, 2.3, n).clip(0, 11).astype(int)
        pause_count        = np.random.normal(12.5, 7.5, n).clip(3, 34).astype(int)
        total_pause_time   = np.random.normal(265, 135, n).clip(45, 750)
        focus_time         = np.random.normal(3250, 1050, n).clip(1100, 6400)
        frustration_events = np.random.normal(8.5, 4.5, n).clip(2, 23).astype(int)
        session_count      = np.random.normal(16, 8, n).clip(6, 42).astype(int)
        days_active        = np.random.normal(34, 16, n).clip(12, 95).astype(int)

    elif level == 'Intermediate':
        hints_used         = np.random.normal(4.5, 3, n).clip(0, 16).astype(int)
        correct_answers    = np.random.normal(40, 10, n).clip(22, 72).astype(int)
        wrong_answers      = np.random.normal(6, 3.5, n).clip(1, 18).astype(int)
        objectives_completed = np.random.normal(15, 5, n).clip(7, 29).astype(int)
        knowledge_score    = np.random.normal(74, 9, n).clip(58, 89)
        progression_rate   = np.random.normal(0.77, 0.11, n).clip(0.55, 0.93)
        retry_after_fail   = np.random.normal(3, 2.2, n).clip(0, 11).astype(int)
        load_time          = np.random.normal(1.55, 0.7, n).clip(0.4, 4.2)
        crash_count        = np.random.normal(0.7, 1, n).clip(0, 5).astype(int)
        lag_events         = np.random.normal(3.8, 3, n).clip(0, 14).astype(int)
        frame_drops        = np.random.normal(4.8, 4, n).clip(0, 18).astype(int)
        api_errors         = np.random.normal(1.1, 1.6, n).clip(0, 8).astype(int)
        
        playtime_voluntary = np.random.normal(102, 36, n).clip(45, 210).astype(int)
        bonus_collected    = np.random.normal(29, 12, n).clip(10, 62).astype(int)
        challenges_attempted = np.random.normal(26, 8, n).clip(12, 54).astype(int)
        idle_time          = np.random.normal(82, 48, n).clip(18, 240)
        exploration_rate   = np.random.normal(0.80, 0.09, n).clip(0.58, 0.95)
        combo_count        = np.random.normal(25, 10, n).clip(10, 58).astype(int)
        skip_count         = np.random.normal(3.8, 2.8, n).clip(0, 12).astype(int)
        help_requests      = np.random.normal(4.8, 3, n).clip(0, 14).astype(int)
        give_up_count      = np.random.normal(1.8, 1.8, n).clip(0, 7).astype(int)
        pause_count        = np.random.normal(9, 6, n).clip(2, 24).astype(int)
        total_pause_time   = np.random.normal(175, 105, n).clip(35, 550)
        focus_time         = np.random.normal(4350, 1250, n).clip(1700, 8200)
        frustration_events = np.random.normal(5.5, 3.5, n).clip(1, 17).astype(int)
        session_count      = np.random.normal(24, 10, n).clip(9, 52).astype(int)
        days_active        = np.random.normal(50, 20, n).clip(18, 125).astype(int)

    elif level == 'Advanced':
        hints_used         = np.random.normal(2, 1.8, n).clip(0, 9).astype(int)
        correct_answers    = np.random.normal(56, 11, n).clip(38, 85).astype(int)
        wrong_answers      = np.random.normal(3.5, 2.2, n).clip(0, 11).astype(int)
        objectives_completed = np.random.normal(20, 6, n).clip(12, 33).astype(int)
        knowledge_score    = np.random.normal(85, 6.5, n).clip(74, 96)
        progression_rate   = np.random.normal(0.88, 0.06, n).clip(0.75, 0.97)
        retry_after_fail   = np.random.normal(1.4, 1.3, n).clip(0, 6).astype(int)
        load_time          = np.random.normal(1.05, 0.45, n).clip(0.35, 3.2)
        crash_count        = np.random.normal(0.35, 0.6, n).clip(0, 4).astype(int)
        lag_events         = np.random.normal(2.2, 1.8, n).clip(0, 10).astype(int)
        frame_drops        = np.random.normal(3, 2.5, n).clip(0, 13).astype(int)
        api_errors         = np.random.normal(0.6, 1, n).clip(0, 5).astype(int)
        
        playtime_voluntary = np.random.normal(148, 44, n).clip(75, 270).astype(int)
        bonus_collected    = np.random.normal(44, 13, n).clip(22, 78).astype(int)
        challenges_attempted = np.random.normal(34, 9, n).clip(20, 64).astype(int)
        idle_time          = np.random.normal(50, 32, n).clip(8, 160)
        exploration_rate   = np.random.normal(0.86, 0.07, n).clip(0.70, 0.98)
        combo_count        = np.random.normal(39, 12, n).clip(18, 78).astype(int)
        skip_count         = np.random.normal(1.8, 1.7, n).clip(0, 8).astype(int)
        help_requests      = np.random.normal(2.2, 1.8, n).clip(0, 9).astype(int)
        give_up_count      = np.random.normal(0.7, 0.9, n).clip(0, 4).astype(int)
        pause_count        = np.random.normal(5.5, 3.5, n).clip(1, 16).astype(int)
        total_pause_time   = np.random.normal(105, 65, n).clip(18, 380)
        focus_time         = np.random.normal(5900, 1550, n).clip(3000, 10500)
        frustration_events = np.random.normal(3, 2.2, n).clip(0, 10).astype(int)
        session_count      = np.random.normal(36, 14, n).clip(14, 78).astype(int)
        days_active        = np.random.normal(78, 27, n).clip(28, 155).astype(int)

    else:  # Expert
        hints_used         = np.random.normal(0.8, 1, n).clip(0, 5).astype(int)
        correct_answers    = np.random.normal(72, 11, n).clip(50, 98).astype(int)
        wrong_answers      = np.random.normal(2, 1.6, n).clip(0, 7).astype(int)
        objectives_completed = np.random.normal(23, 6, n).clip(14, 37).astype(int)
        knowledge_score    = np.random.normal(93, 4.5, n).clip(84, 99.5)
        progression_rate   = np.random.normal(0.95, 0.03, n).clip(0.87, 0.99)
        retry_after_fail   = np.random.normal(0.7, 0.9, n).clip(0, 4).astype(int)
        load_time          = np.random.normal(0.85, 0.4, n).clip(0.3, 2.3)
        crash_count        = np.random.normal(0.2, 0.5, n).clip(0, 3).astype(int)
        lag_events         = np.random.normal(1.4, 1.4, n).clip(0, 7).astype(int)
        frame_drops        = np.random.normal(2.1, 2, n).clip(0, 9).astype(int)
        api_errors         = np.random.normal(0.4, 0.8, n).clip(0, 4).astype(int)
        
        playtime_voluntary = np.random.normal(182, 48, n).clip(95, 360).astype(int)
        bonus_collected    = np.random.normal(55, 14, n).clip(30, 92).astype(int)
        challenges_attempted = np.random.normal(40, 10, n).clip(24, 72).astype(int)
        idle_time          = np.random.normal(30, 20, n).clip(5, 110)
        exploration_rate   = np.random.normal(0.90, 0.06, n).clip(0.75, 0.99)
        combo_count        = np.random.normal(53, 14, n).clip(28, 95).astype(int)
        skip_count         = np.random.normal(0.9, 1.2, n).clip(0, 5).astype(int)
        help_requests      = np.random.normal(1.3, 1.4, n).clip(0, 6).astype(int)
        give_up_count      = np.random.normal(0.3, 0.6, n).clip(0, 3).astype(int)
        pause_count        = np.random.normal(4, 3, n).clip(0, 12).astype(int)
        total_pause_time   = np.random.normal(60, 48, n).clip(8, 240)
        focus_time         = np.random.normal(7500, 1850, n).clip(4000, 15500)
        frustration_events = np.random.normal(1.6, 1.6, n).clip(0, 7).astype(int)
        session_count      = np.random.normal(47, 17, n).clip(20, 105).astype(int)
        days_active        = np.random.normal(95, 30, n).clip(40, 190).astype(int)

    # === Device & Screen ===
    device_probs = {'desktop': 0.52, 'tablet': 0.28, 'mobile': 0.20}
    devices = np.random.choice(list(device_probs.keys()), n, p=list(device_probs.values()))

    screen_width = np.where(devices == 'desktop', 
                          np.random.choice([1920, 2560, 1366, 1440], n),
                          np.where(devices == 'tablet',
                                   np.random.choice([1080, 1200, 800], n),
                                   np.random.choice([720, 1080, 1440], n)))

    screen_height = np.where(devices == 'desktop',
                           np.random.choice([1080, 1440, 768, 900], n),
                           np.where(devices == 'tablet',
                                    np.random.choice([1920, 1600, 1280], n),
                                    np.random.choice([1520, 2400, 1280], n)))

    data = {
        'hints_used': hints_used,
        'correct_answers': correct_answers,
        'wrong_answers': wrong_answers,
        'objectives_completed': objectives_completed,
        'knowledge_score': np.round(knowledge_score, 2),
        'progression_rate': np.round(progression_rate, 2),
        'retry_after_fail': retry_after_fail,
        'load_time': np.round(load_time, 2),
        'crash_count': crash_count,
        'lag_events': lag_events,
        'frame_drops': frame_drops,
        'api_errors': api_errors,
        'device_type': devices,
        'screen_width': screen_width,
        'screen_height': screen_height,
        'playtime_voluntary': playtime_voluntary,
        'bonus_collected': bonus_collected,
        'challenges_attempted': challenges_attempted,
        'idle_time': np.round(idle_time, 2),
        'exploration_rate': np.round(exploration_rate, 2),
        'combo_count': combo_count,
        'skip_count': skip_count,
        'help_requests': help_requests,
        'give_up_count': give_up_count,
        'pause_count': pause_count,
        'total_pause_time': np.round(total_pause_time, 2),
        'focus_time': np.round(focus_time, 2),
        'frustration_events': frustration_events,
        'session_count': session_count,
        'days_active': days_active
    }
    return pd.DataFrame(data)


# ================== GÉNÉRATION ==================
print("Génération du dataset équilibré en cours...")
dfs = []
for i, level in enumerate(labels):
    print(f"→ {sizes[i]} {level}")
    df = generate_player_data(sizes[i], level)
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

full_df.to_csv('players_dataset_18000_balanced_6profiles.csv', index=False)

print(f"\n✅ Dataset 100% équilibré généré avec succès !")
print(f"   Total lignes : {len(full_df)}")
print(f"   Par profil   : 3000 lignes")
print(f"   Fichier      : players_dataset_18000_balanced_6profiles.csv")
print("\nDistribution :")
print(full_df.groupby('player_level').size() if 'player_level' in full_df.columns else "Player_level non inclus (ajoutable si besoin)")
print("\nAperçu :")
print(full_df.head(3))