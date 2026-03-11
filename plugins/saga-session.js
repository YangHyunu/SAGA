//@name saga_session
//@display-name SAGA Session + State Bridge
//@api 3.0
//@version 2.0.0

// RisuAI Plugin v3.0 — SAGA Session Identifier + Scriptstate Bridge
// Injects a sentinel message so SAGA can:
//   1. Distinguish chat branches (session ID)
//   2. Receive scriptstate variables for live_state.md (HP, location, etc.)

(async () => {
  try {
    Risuai.addRisuReplacer('beforeRequest', async (messages, type) => {
      const char = await Risuai.getCharacter();
      if (!char) return messages;

      const chatIdx = await Risuai.getCurrentChatIndex();
      const chat = char.chats?.[chatIdx];
      const chaId = (char.chaId || '').slice(0, 8);
      const chatId = (chat?.id || '').slice(0, 8);
      const isGroup = char.type === 'group' ? '1' : '0';

      // Build sentinel: line 1 = session metadata
      let sentinel = `@@SAGA:sid=${chaId}-${chatId}&grp=${isGroup}`;

      // Line 2 = scriptstate (if any variables exist)
      const scriptstate = chat?.scriptstate;
      if (scriptstate && typeof scriptstate === 'object') {
        const keys = Object.keys(scriptstate);
        if (keys.length > 0) {
          sentinel += `\n@@SAGA_STATE:${JSON.stringify(scriptstate)}`;
        }
      }

      messages.unshift({ role: 'system', content: sentinel });
      return messages;
    });

    console.log('SAGA Session + State Bridge initialized');
  } catch (error) {
    console.log(`SAGA plugin error: ${error.message}`);
  }
})();
