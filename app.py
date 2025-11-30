import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import time
import pandas as pd
import os

# --- Page Config ---
st.set_page_config(page_title="Smart Office RL Control Viz", layout="wide", page_icon="🏢")

# --- CSS for Custom Visuals ---
st.markdown("""
<style>
    .office-container {
        border: 4px solid #333;
        border-radius: 10px;
        padding: 20px;
        height: 400px;
        position: relative;
        transition: background-color 0.5s ease;
        overflow: hidden;
        background-size: cover;
        font-family: sans-serif;
    }
    .office-floor {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 50px;
        background-color: #6D4C41;
        border-top: 2px solid #4E342E;
        z-index: 1;
    }
    .window-container {
        position: absolute;
        top: 30px;
        right: 10px;
        width: 120px;
        height: 150px;
        background-color: #87CEEB;
        border: 4px solid #555;
        text-align: center;
        z-index: 0;
    }
    .window-frame-inner {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 40px;
    }
    .hvac-unit {
        position: absolute;
        top: 20px;
        left: 20px;
        width: 110px;
        height: 70px;
        background-color: #ddd;
        border: 2px solid #999;
        border-radius: 4px;
        text-align: center;
        padding-top: 5px;
        font-weight: bold;
        z-index: 2;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .hvac-airflow {
        font-size: 24px;
        margin-top: 2px;
    }
    .purifier-unit {
        position: absolute;
        bottom: 60px;
        left: 140px;
        width: 70px;
        height: 90px;
        border: 2px solid #555;
        border-radius: 5px;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 3;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        font-size: 11px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .purifier-icon {
        font-size: 35px;
    }
    .occupant {
        position: absolute;
        bottom: 55px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 90px;
        text-shadow: 2px 2px 4px #000000;
        z-index: 5;
    }
    .co2-cloud {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 100px;
        opacity: 0.3;
        color: #555;
        z-index: 4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ENVIRONMENT & AGENT
# ==========================================

class OfficeCoControlEnv(gym.Env):
    def __init__(self):
        super(OfficeCoControlEnv, self).__init__()
        # Constants
        self.TIME_STEP = 0.25  
        self.MAX_EPISODE_STEPS = 200 
        self.T_OPTIMAL = 24.0
        self.CO2_THRESHOLD = 1000.0
        self.PM25_REF = 15.0 # Target max PM2.5
        self.THERMAL_MASS = 50.0 
        self.HEAT_TRANSFER_COEFF = 2.0 
        self.MAX_OCCUPANCY_HEAT = 0.5
        self.MAX_CO2_GEN = 25.0
        self.PM25_INDOOR_SOURCE = 2.0 
        self.HVAC_POWER_MAX = 3.0 
        self.PURIFIER_POWER_MAX = 0.15 
        self.WINDOW_POWER_MAX = 0.05 
        
        # Normalization Values for Rewards
        self.NORM_ENERGY = self.HVAC_POWER_MAX + self.PURIFIER_POWER_MAX + self.WINDOW_POWER_MAX
        self.NORM_THERMAL = 100.0 # Deviation squared divisor
        self.NORM_CO2 = 4000.0    # Excess CO2 divisor
        self.NORM_PM25 = 100.0    # Max expected PM2.5 for normalization

        # Spaces
        self.state_high = np.array([2000.0, 40.0, 200.0, 40.0, 200.0, 1.0], dtype=np.float32)
        self.state_low = np.array([400.0, 15.0, 0.0, 10.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.state = None
        self.current_step = 0

    def _get_obs(self):
        obs = (self.state - self.state_low) / (self.state_high - self.state_low)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        T_out = np.random.uniform(20.0, 30.0)
        PM25_out = np.random.uniform(5.0, 30.0)
        # State: [CO2, T_in, PM2.5, T_out, PM2.5_out, Occ]
        self.state = np.array([450.0, 24.0, 12.0, T_out, PM25_out, 1.0], dtype=np.float32)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        occ = 1.0 
        
        # Actions
        hvac_action = float(np.clip(action[0], -1.0, 1.0))
        purifier_action = float(np.clip(action[1], 0.0, 1.0))
        window_action = float(np.clip(action[2], 0.0, 1.0))

        C_CO2, T_in, C_PM25, T_out, C_PM25_out, _ = self.state

        # --- PHYSICS ---
        q_hvac = hvac_action * self.HVAC_POWER_MAX
        q_load = (occ * self.MAX_OCCUPANCY_HEAT) + self.HEAT_TRANSFER_COEFF * (T_out - T_in) * 0.05
        T_in_new = T_in + (q_hvac + q_load) / self.THERMAL_MASS

        vent_factor = 0.05 + (window_action * 0.5)
        
        # CO2 Physics
        co2_gen = occ * self.MAX_CO2_GEN
        C_CO2_new = C_CO2 + co2_gen - (vent_factor * (C_CO2 - 420.0))

        # PM2.5 Physics
        pm_gen = self.PM25_INDOOR_SOURCE
        # Purifier efficiency 0.8. Vent brings in outdoor PM.
        pm_removal = purifier_action * 0.8 * C_PM25 
        pm_exchange = vent_factor * (C_PM25_out - C_PM25)
        C_PM25_new = C_PM25 + pm_gen - pm_removal + pm_exchange

        self.state = np.array([
            np.clip(C_CO2_new, 400, 5000),
            np.clip(T_in_new, 10, 45),
            np.clip(C_PM25_new, 0, 500),
            T_out + np.random.normal(0, 0.1),
            C_PM25_out + np.random.normal(0, 0.5),
            occ
        ], dtype=np.float32)

        # --- UPDATED REWARD FUNCTION ---
        
        # 1. Energy Cost (Normalized 0-1)
        raw_energy = abs(hvac_action * self.HVAC_POWER_MAX) + (purifier_action * self.PURIFIER_POWER_MAX) + (window_action * self.WINDOW_POWER_MAX)
        norm_energy = raw_energy / self.NORM_ENERGY

        # 2. Thermal Comfort (Squared deviation)
        raw_thermal_dev = (T_in_new - self.T_OPTIMAL) ** 2
        norm_thermal = min(1.0, raw_thermal_dev / self.NORM_THERMAL)

        # 3. CO2 Penalty (Linear excess)
        raw_co2_excess = max(0, C_CO2_new - self.CO2_THRESHOLD)
        norm_co2 = min(1.0, raw_co2_excess / self.NORM_CO2)

        # 4. PM2.5 Penalty (FIXED: Strict Linear Scaling)
        # Previous exponential was too weak. 
        # Now: if PM2.5 > 15, we penalize strictly based on ratio to 100.
        # If PM2.5 is 50, penalty is 0.5.
        raw_pm_excess = max(0, C_PM25_new - self.PM25_REF)
        norm_pm = min(1.0, raw_pm_excess / (self.NORM_PM25 - self.PM25_REF))

        # Weights
        w_energy = 0.4
        w_thermal = 0.8
        w_co2 = 0.8
        w_pm = 1.2  # Increased priority so it learns to use purifier

        penalty_sum = (w_energy * norm_energy) + (w_thermal * norm_thermal) + (w_co2 * norm_co2) + (w_pm * norm_pm)
        reward = 1.0 - penalty_sum

        terminated = False
        truncated = self.current_step >= self.MAX_EPISODE_STEPS
        
        info = {
            "T_in": T_in_new, "CO2": C_CO2_new, "PM2.5": C_PM25_new, "Power": raw_energy,
            "HVAC": hvac_action, "Window": window_action, "Purifier": purifier_action, 
            "T_out": T_out, "Occupancy": occ
        }
        return self._get_obs(), reward, terminated, truncated, info

# --- DDPG Components (Standard) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done, dtype=np.float32)).unsqueeze(1).to(device)
        )
    def __len__(self): return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )
    def forward(self, state): return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state, action): return self.net(torch.cat([state, action], 1))

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(100000)
        self.mse = nn.MSELoss()
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005

    def get_action(self, state, noise=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        if noise > 0: action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def update(self):
        if len(self.replay_buffer) < self.batch_size: return
        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            target_q = self.critic_target(ns, self.actor_target(ns))
            target_val = r + (1 - d) * self.gamma * target_q
        current_q = self.critic(s, a)
        critic_loss = self.mse(current_q, target_val)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        for tp, lp in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(self.tau * lp.data + (1 - self.tau) * tp.data)
        for tp, lp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * lp.data + (1 - self.tau) * tp.data)
            
    def save_checkpoint(self, filename="checkpoint"):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load_checkpoint(self, filename="checkpoint"):
        if os.path.exists(filename + "_actor.pth"):
            self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
            self.critic_target.load_state_dict(self.critic.state_dict())
            return True
        return False

# ==========================================
# 2. VISUALIZATION HELPER (FIXED)
# ==========================================
def render_office_visual(info):
    t_in = info['T_in']
    hvac_val = info['HVAC']
    win_val = info['Window']
    pur_val = info['Purifier']
    occupancy = info['Occupancy']
    co2 = info['CO2']

    # 1. Background Color
    if t_in < 22.0:
        ratio = max(0, (t_in - 15.0) / (22.0 - 15.0))
        r = int(173 + (144 - 173) * ratio)
        g = int(216 + (238 - 216) * ratio)
        b = int(230 + (144 - 230) * ratio)
        bg_color = f"rgb({r},{g},{b})"
    elif t_in > 26.0:
        ratio = min(1, (t_in - 26.0) / (35.0 - 26.0))
        r = int(144 + (240 - 144) * ratio)
        g = int(238 + (128 - 238) * ratio)
        b = int(144 + (128 - 144) * ratio)
        bg_color = f"rgb({r},{g},{b})"
    else:
        bg_color = "#90ee90" 

    # 2. Window State
    window_icon = "🪟" if win_val > 0.1 else "🔒"
    window_style = "opacity: 1.0;" if win_val > 0.1 else "opacity: 0.6; background-color: #ccc;"

    # 3. HVAC State
    if hvac_val > 0.1:
        hvac_icon = "🔥 Heating"
        airflow_icon = "♨️"
        hvac_color = "#ffcccb" 
    elif hvac_val < -0.1:
        hvac_icon = "❄️ Cooling"
        airflow_icon = "💨"
        hvac_color = "#add8e6" 
    else:
        hvac_icon = "⚪ Off"
        airflow_icon = ""
        hvac_color = "#ddd"

    # 4. Purifier State
    if pur_val > 0.1:
        pur_icon = "🌀"
        pur_bg = "#E0F7FA" 
        pur_status = "ON"
    else:
        pur_icon = "⚪"
        pur_bg = "#EEE"
        pur_status = "OFF"

    # 5. Occupancy
    human_html = "👤" * int(occupancy) if occupancy >= 0.5 else ""
    
    # 6. CO2 Visual
    co2_html = "☁️" if co2 > 1200 else ""

    # FIXED: Using strict string concatenation to avoid f-string newline breaks
    html = (
        f'<div class="office-container" style="background-color: {bg_color};">'
        f'<div class="hvac-unit" style="background-color: {hvac_color};">{hvac_icon}<div class="hvac-airflow">{airflow_icon}</div></div>'
        f'<div class="window-container" style="{window_style}"><div class="window-frame-inner">{window_icon}</div><div style="font-size: 12px;">Open: {win_val:.0%}</div></div>'
        f'<div class="co2-cloud">{co2_html}</div>'
        f'<div class="occupant">{human_html}</div>'
        f'<div class="purifier-unit" style="background-color: {pur_bg};"><div class="purifier-icon">{pur_icon}</div><div>{pur_status}</div></div>'
        f'<div class="office-floor"></div>'
        '</div>'
    )
    return html

# ==========================================
# 3. STREAMLIT INTERFACE MAIN
# ==========================================

if 'agent' not in st.session_state:
    st.session_state['agent'] = DDPGAgent(6, 3)
if 'trained' not in st.session_state:
    st.session_state['trained'] = False
if 'training_rewards' not in st.session_state:
    st.session_state['training_rewards'] = []

# --- Sidebar ---
st.sidebar.title("🎮 Control Panel")
st.sidebar.markdown("### Model Management")

col_save, col_load = st.sidebar.columns(2)
if col_save.button("💾 Save Model"):
    if st.session_state['trained']:
        st.session_state['agent'].save_checkpoint("ddpg_office")
        st.sidebar.success("Model saved!")
    else:
        st.sidebar.warning("Train first!")

if col_load.button("📂 Load Model"):
    success = st.session_state['agent'].load_checkpoint("ddpg_office")
    if success:
        st.session_state['trained'] = True
        st.sidebar.success("Model loaded!")
    else:
        st.sidebar.error("No saved model.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Training Config")
train_episodes = st.sidebar.slider("Training Episodes", 10, 1000, 100, step=10)
train_btn = st.sidebar.button("Start Training", type="primary")

if st.sidebar.button("Reset Agent"):
    st.session_state['agent'] = DDPGAgent(6, 3)
    st.session_state['trained'] = False
    st.session_state['training_rewards'] = []
    st.sidebar.success("Agent reset!")

# --- Main ---
st.title("🏢 Smart Office RL Controller")
st.caption("Deep Deterministic Policy Gradient (DDPG) Agent optimizing Comfort & Energy")

tab1, tab2 = st.tabs(["📈 Training Process", "🖥️ Real-time Office Simulation"])

# TAB 1: Training
with tab1:
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    chart_placeholder = st.empty()

    if train_btn:
        st.session_state['training_rewards'] = []
        env = OfficeCoControlEnv()
        agent = st.session_state['agent']
        
        status_placeholder.info("🚀 Training initialized...")
        rewards_list = []
        plot_data = pd.DataFrame(columns=["Reward", "Moving_Avg"])
        
        progress_bar = progress_placeholder.progress(0)
        
        for ep in range(train_episodes):
            state, _ = env.reset()
            ep_reward = 0
            for step in range(200):
                noise = 0.2 if ep < (train_episodes * 0.3) else 0.05
                action = agent.get_action(state, noise)
                next_state, reward, term, trunc, info = env.step(action)
                done = term or trunc
                agent.replay_buffer.push(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                ep_reward += reward
                if done: break
            
            rewards_list.append(ep_reward)
            if (ep + 1) % 5 == 0 or ep == 0:
                avg = np.mean(rewards_list[-20:]) if len(rewards_list) >= 20 else np.mean(rewards_list)
                status_placeholder.markdown(f"**Episode {ep+1}/{train_episodes}** | Last Reward: `{ep_reward:.2f}` | Avg (20): `{avg:.2f}`")
                progress_bar.progress((ep + 1) / train_episodes)
                new_row = pd.DataFrame({"Reward": [ep_reward],"Moving_Avg": [avg]})
                plot_data = pd.concat([plot_data, new_row], ignore_index=True)
                chart_placeholder.line_chart(plot_data)

        st.session_state['trained'] = True
        st.session_state['training_rewards'] = rewards_list
        status_placeholder.success("✅ Training Complete!")
        progress_placeholder.empty()
    
    elif st.session_state['trained'] and st.session_state['training_rewards']:
         st.info("Showing results from previous training session.")
         rewards = st.session_state['training_rewards']
         window = 20
         moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
         padded_ma = np.concatenate((np.full(window-1, np.nan), moving_avg))
         df = pd.DataFrame({"Episode Reward": rewards, "Moving Average": padded_ma})
         chart_placeholder.line_chart(df)
         
    else:
         st.warning("⚠️ **Model not trained yet.**")
         st.markdown("1. Adjust **Training Episodes** in the sidebar.\n2. Click **Start Training**.\n3. Or click **Load Model**.")

# TAB 2: Simulation
with tab2:
    col_visual, col_stats = st.columns([3, 2])
    
    with col_stats:
            st.markdown("### Simulation Controls")
            sim_speed = st.slider("Speed (delay in sec)", 0.0, 1.0, 0.1)
            sim_steps = st.number_input("Steps to Run", 50, 500, 200)
            run_sim = st.button("▶️ Start Live Simulation", type="primary")
            
            st.markdown("---")
            st.markdown("### Live Statistics")
            stat_box_indoor = st.empty()
            stat_box_outdoor = st.empty()
            stat_box_power = st.empty()

    with col_visual:
            st.markdown("### Office Environment Visualizer")
            visual_placeholder = st.empty()
            visual_placeholder.markdown("""
            <div class="office-container" style="background-color: #eee; display:flex; justify-content:center; align-items:center; flex-direction:column; color:#555;">
                <h2 style="margin:0;">Waiting to Start</h2>
                <p>Click 'Start Live Simulation' on the right.</p>
            </div>""", unsafe_allow_html=True)

    if run_sim:
        if not st.session_state['trained']:
            st.error("⚠️ Please train the agent in Tab 1 or Load a Model first.")
        else:
            sim_env = OfficeCoControlEnv()
            state, _ = sim_env.reset()
            sim_agent = st.session_state['agent']
            sim_agent.actor.eval()
            
            for t in range(sim_steps):
                action = sim_agent.get_action(state, noise=0.0)
                state, reward, done, trunc, info = sim_env.step(action)
                
                # Update Visual
                html_content = render_office_visual(info)
                visual_placeholder.markdown(html_content, unsafe_allow_html=True)
                
                # Update Stats
                with stat_box_indoor.container():
                        st.markdown('<div class="metric-container"><b>🏠 Indoor Air</b></div>', unsafe_allow_html=True)
                        k1, k2, k3 = st.columns(3)
                        t_val = info['T_in']
                        t_delta = round(t_val - 24.0, 1)
                        k1.metric("Temp (°C)", f"{t_val:.1f}", delta=t_delta, delta_color="inverse")
                        
                        co2_val = info['CO2']
                        co2_delta = round(co2_val - 1000, 0)
                        k2.metric("CO2 (ppm)", f"{int(co2_val)}", delta=-co2_delta, delta_color="normal")
                        k3.metric("PM2.5", f"{info['PM2.5']:.1f}")

                with stat_box_outdoor.container():
                        st.markdown('<div class="metric-container" style="margin-top: 10px;"><b>🌳 Outdoor Conditions</b></div>', unsafe_allow_html=True)
                        o1, o2 = st.columns(2)
                        o1.metric("Out Temp", f"{info['T_out']:.1f}°C")
                        o2.metric("Out PM2.5", f"{state[4]:.1f}")

                with stat_box_power.container():
                        st.markdown('<div class="metric-container" style="margin-top: 10px;"><b>⚡ Energy & Reward</b></div>', unsafe_allow_html=True)
                        p1, p2 = st.columns(2)
                        p1.metric("Total Power", f"{info['Power']:.2f} kW")
                        p2.metric("Instant Reward", f"{reward:.3f}")
                
                if sim_speed > 0:
                        time.sleep(sim_speed)
                if done: 
                    break
            st.success("Simulation finished.")