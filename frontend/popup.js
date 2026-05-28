/* ============================================================
   popup.js — YT Sentiment Analyzer Extension
   Handles: sentiment analysis, GSAP animations, UI state
   ============================================================ */

'use strict';

/* ─────────────────────────────────────────────
   1. SENTIMENT ENGINE
   Simple keyword-weighted classifier that mimics
   the LightGBM + TF-IDF model's behavior.
   ───────────────────────────────────────────── */

const POSITIVE_WORDS = [
    'amazing', 'awesome', 'excellent', 'great', 'love', 'loved', 'best',
    'fantastic', 'wonderful', 'brilliant', 'perfect', 'incredible', 'outstanding',
    'superb', 'beautiful', 'happy', 'enjoy', 'enjoyed', 'helpful', 'good',
    'nice', 'cool', 'impressive', 'thanks', 'thank', 'appreciate', 'liked',
    'like', 'well', 'helpful', 'insightful', 'informative', 'fun', 'interesting',
    'clear', 'easy', 'elegant', 'genius', 'legend', 'goat', 'fire', 'banger',
    'underrated', 'flawless', 'smooth', 'clean', 'wow', 'satisfying', 'solid',
    'recommend', 'subscribed', 'subscribe', 'support', 'masterpiece', 'top',
    'god', 'wholesome', 'blessed', 'inspiring', 'motivated', 'excited'
];

const NEGATIVE_WORDS = [
    'bad', 'terrible', 'horrible', 'awful', 'worst', 'hate', 'hated', 'useless',
    'boring', 'waste', 'disappointed', 'disappointing', 'ugly', 'broken',
    'frustrating', 'annoying', 'stupid', 'dumb', 'ridiculous', 'pathetic',
    'trash', 'garbage', 'toxic', 'disgusting', 'fail', 'failed', 'poor',
    'overrated', 'dislike', 'unliked', 'stop', 'quit', 'unsubscribe',
    'misleading', 'wrong', 'incorrect', 'error', 'bug', 'crash', 'slow',
    'clickbait', 'spam', 'scam', 'fake', 'lie', 'lying', 'lied', 'idiot',
    'loser', 'cringe', 'sad', 'sick', 'mess', 'awful', 'nonsense', 'brainwash'
];

const NEGATION_WORDS = ['not', 'no', 'never', 'neither', 'nor', 'nothing', 'nobody', "n't", 'without'];
const INTENSIFIERS  = ['very', 'really', 'absolutely', 'extremely', 'totally', 'so', 'super', 'incredibly'];

/**
 * Tokenize text: lowercase, strip punctuation, split on whitespace.
 * @param {string} text
 * @returns {string[]}
 */
function tokenize(text) {
    return text
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter(Boolean);
}

/**
 * Run a simplified TF-IDF-style weighted scoring to predict sentiment.
 * Returns { label: 'positive'|'neutral'|'negative', scores: {positive, neutral, negative} }
 * @param {string} comment
 * @returns {{ label: string, scores: {positive: number, neutral: number, negative: number} }}
 */
function analyzeSentiment(comment) {
    const tokens = tokenize(comment);
    let posScore = 0;
    let negScore = 0;
    let negated  = false;

    for (let i = 0; i < tokens.length; i++) {
        const word = tokens[i];
        const prevWord = tokens[i - 1] || '';
        const prevPrev = tokens[i - 2] || '';

        // Detect negation window (2 words look-back)
        if (NEGATION_WORDS.includes(word)) {
            negated = true;
            continue;
        }

        // Reset negation after 3 tokens
        if (i > 0 && !NEGATION_WORDS.includes(prevWord) && !NEGATION_WORDS.includes(prevPrev)) {
            negated = false;
        }

        // Intensifier multiplier
        const intensified = INTENSIFIERS.includes(prevWord) ? 1.6 : 1.0;

        if (POSITIVE_WORDS.includes(word)) {
            const delta = 1.0 * intensified;
            negated ? negScore += delta : posScore += delta;
        }

        if (NEGATIVE_WORDS.includes(word)) {
            const delta = 1.0 * intensified;
            negated ? posScore += delta * 0.5 : negScore += delta;
        }
    }

    // Tiny base score so neither class is ever 0
    posScore += 0.15;
    negScore += 0.15;
    const neuScore = 0.5;

    const total = posScore + neuScore + negScore;

    const scores = {
        positive: posScore / total,
        neutral:  neuScore / total,
        negative: negScore / total,
    };

    // Label = argmax
    let label = 'neutral';
    if (scores.positive > scores.negative && scores.positive > scores.neutral) label = 'positive';
    if (scores.negative > scores.positive && scores.negative > scores.neutral) label = 'negative';

    return { label, scores };
}


/* ─────────────────────────────────────────────
   2. DOM ELEMENT REFERENCES
   ───────────────────────────────────────────── */

const elLiveDot     = document.getElementById('live-dot');
const elLiveLabel   = document.getElementById('live-label');

const screenNotYT   = document.getElementById('screen-not-yt');
const screenLoading = document.getElementById('screen-loading');
const screenResults = document.getElementById('screen-results');
const screenManual  = document.getElementById('screen-manual');

// Donut
const elDonutTotal  = document.getElementById('donut-total');
const arcPos        = document.getElementById('arc-pos');
const arcNeu        = document.getElementById('arc-neu');
const arcNeg        = document.getElementById('arc-neg');

// Legend counts & pct
const elPosCount = document.getElementById('pos-count');
const elNeuCount = document.getElementById('neu-count');
const elNegCount = document.getElementById('neg-count');
const elPosPct   = document.getElementById('pos-pct');
const elNeuPct   = document.getElementById('neu-pct');
const elNegPct   = document.getElementById('neg-pct');

// Bar strips
const barPos  = document.getElementById('bar-pos');
const barNeu  = document.getElementById('bar-neu');
const barNeg  = document.getElementById('bar-neg');
const bpvPos  = document.getElementById('bpv-pos');
const bpvNeu  = document.getElementById('bpv-neu');
const bpvNeg  = document.getElementById('bpv-neg');

// Comment list
const commentList = document.getElementById('comment-list');

// Filter tabs
const filterTabs = document.querySelectorAll('.tab');

// Bottom nav
const bnavAnalyze = document.getElementById('bnav-analyze');
const bnavManual  = document.getElementById('bnav-manual');

// Manual mode
const manualTextarea  = document.getElementById('manual-textarea');
const manualAnalyzeBtn = document.getElementById('manual-analyze-btn');
const manualResult    = document.getElementById('manual-result');
const manualBadge     = document.getElementById('manual-badge');
const manualConfBars  = document.getElementById('manual-conf-bars');

// Donut circumference (r=46 → 2πr ≈ 289)
const CIRCUMFERENCE = 2 * Math.PI * 46;  // ≈ 289


/* ─────────────────────────────────────────────
   3. STATE
   ───────────────────────────────────────────── */

/** @type {{ text: string, label: string, scores: object }[]} */
let allComments = [];
let activeFilter = 'all';
let currentScreen = 'analyze';   // 'analyze' | 'manual'


/* ─────────────────────────────────────────────
   4. SCREEN MANAGEMENT
   ───────────────────────────────────────────── */

/**
 * Show one state screen, hide the others.
 * @param {'not-yt'|'loading'|'results'|'manual'} name
 */
function showScreen(name) {
    screenNotYT.classList.add('hidden');
    screenLoading.classList.add('hidden');
    screenResults.classList.add('hidden');
    screenManual.classList.add('hidden');

    const map = {
        'not-yt':  screenNotYT,
        'loading': screenLoading,
        'results': screenResults,
        'manual':  screenManual,
    };

    const el = map[name];
    if (el) el.classList.remove('hidden');
}


/* ─────────────────────────────────────────────
   5. DONUT CHART ANIMATION
   ───────────────────────────────────────────── */

/**
 * Animate the 3-segment donut chart using GSAP.
 * Each segment is offset so they sit next to each other on the ring.
 * @param {number} posRatio  0–1
 * @param {number} neuRatio  0–1
 * @param {number} negRatio  0–1
 */
function animateDonut(posRatio, neuRatio, negRatio) {
    const C = CIRCUMFERENCE;   // full ring length
    const GAP = 6;             // small gap between segments

    const posLen = posRatio * C;
    const neuLen = neuRatio * C;
    const negLen = negRatio * C;

    // --- Positive arc ---
    // dash = posLen, then rest hidden; starts at 0°
    gsap.fromTo('#arc-pos',
        { attr: { 'stroke-dashoffset': C } },
        {
            attr: { 'stroke-dasharray': `${Math.max(posLen - GAP, 0)} ${C}`, 'stroke-dashoffset': 0 },
            duration: 1.1,
            ease: 'power3.out',
        }
    );

    // --- Neutral arc ---
    // starts where positive ends
    const neuOffset = -(posLen);
    gsap.fromTo('#arc-neu',
        { attr: { 'stroke-dashoffset': C } },
        {
            attr: { 'stroke-dasharray': `${Math.max(neuLen - GAP, 0)} ${C}`, 'stroke-dashoffset': neuOffset },
            duration: 1.1,
            ease: 'power3.out',
            delay: 0.08,
        }
    );

    // --- Negative arc ---
    const negOffset = -(posLen + neuLen);
    gsap.fromTo('#arc-neg',
        { attr: { 'stroke-dashoffset': C } },
        {
            attr: { 'stroke-dasharray': `${Math.max(negLen - GAP, 0)} ${C}`, 'stroke-dashoffset': negOffset },
            duration: 1.1,
            ease: 'power3.out',
            delay: 0.16,
        }
    );
}


/* ─────────────────────────────────────────────
   6. ANIMATED COUNTER (GSAP)
   ───────────────────────────────────────────── */

/**
 * Animate a number counter from 0 → target using GSAP.
 * @param {HTMLElement} el
 * @param {number} target
 * @param {string} [suffix='']
 * @param {number} [decimals=0]
 */
function animateCounter(el, target, suffix = '', decimals = 0) {
    const obj = { val: 0 };
    gsap.to(obj, {
        val: target,
        duration: 1.1,
        ease: 'power2.out',
        onUpdate() {
            el.textContent = obj.val.toFixed(decimals) + suffix;
        },
    });
}


/* ─────────────────────────────────────────────
   7. RENDER RESULTS UI
   ───────────────────────────────────────────── */

/**
 * Given a full analysis result, populate every UI element.
 * @param {{ text: string, label: string, scores: object }[]} results
 */
function renderResults(results) {
    allComments = results;

    const total   = results.length;
    const posArr  = results.filter(r => r.label === 'positive');
    const neuArr  = results.filter(r => r.label === 'neutral');
    const negArr  = results.filter(r => r.label === 'negative');

    const posRatio = total ? posArr.length / total : 0;
    const neuRatio = total ? neuArr.length / total : 0;
    const negRatio = total ? negArr.length / total : 0;

    // ── Total counter ──
    animateCounter(elDonutTotal, total);

    // ── Legend counts ──
    animateCounter(elPosCount, posArr.length);
    animateCounter(elNeuCount, neuArr.length);
    animateCounter(elNegCount, negArr.length);

    // ── Legend percentages ──
    gsap.delayedCall(0.3, () => {
        elPosPct.textContent = (posRatio * 100).toFixed(1) + '%';
        elNeuPct.textContent = (neuRatio * 100).toFixed(1) + '%';
        elNegPct.textContent = (negRatio * 100).toFixed(1) + '%';
    });

    // ── Bar strips ──
    gsap.to(barPos, { width: (posRatio * 100) + '%', duration: 1.1, ease: 'power3.out', delay: 0.1 });
    gsap.to(barNeu, { width: (neuRatio * 100) + '%', duration: 1.1, ease: 'power3.out', delay: 0.18 });
    gsap.to(barNeg, { width: (negRatio * 100) + '%', duration: 1.1, ease: 'power3.out', delay: 0.26 });

    bpvPos.textContent = (posRatio * 100).toFixed(0) + '%';
    bpvNeu.textContent = (neuRatio * 100).toFixed(0) + '%';
    bpvNeg.textContent = (negRatio * 100).toFixed(0) + '%';

    // ── Donut ──
    animateDonut(posRatio, neuRatio, negRatio);

    // ── Comment list ──
    renderCommentList(results);

    // ── Animate results section in ──
    showScreen('results');
    gsap.fromTo(screenResults,
        { opacity: 0, y: 10 },
        { opacity: 1, y: 0, duration: 0.45, ease: 'power2.out' }
    );
}


/* ─────────────────────────────────────────────
   8. COMMENT LIST RENDERING
   ───────────────────────────────────────────── */

/** Emoji map for each label */
const EMOJI = { positive: '😊', neutral: '😐', negative: '😞' };
/** Color class for avatar/pill */
const COLOR_CLASS = { positive: 'green', neutral: 'amber', negative: 'red' };

/**
 * Build a single comment card DOM element.
 * @param {{ text: string, label: string, scores: object }} item
 * @param {number} idx
 * @returns {HTMLElement}
 */
function buildCommentCard(item, idx) {
    const color  = COLOR_CLASS[item.label];
    const emoji  = EMOJI[item.label];
    const pct    = (item.scores[item.label] * 100).toFixed(0);

    // Shorten text for display
    const shortText = item.text.length > 110
        ? item.text.slice(0, 110).trimEnd() + '…'
        : item.text;

    // Avatar: first letter of comment (fallback '#')
    const letter = (item.text[0] || '#').toUpperCase();

    const card = document.createElement('div');
    card.className = 'comment-card';
    card.dataset.label = item.label;

    card.innerHTML = `
        <div class="comment-avatar avatar-${color}">${letter}</div>
        <div class="comment-body">
            <div class="comment-top">
                <span class="comment-text">${escapeHtml(shortText)}</span>
                <span class="sentiment-pill pill-${item.label.slice(0,3)}">${emoji} ${capitalise(item.label)}</span>
            </div>
            <div class="conf-micro-track">
                <div class="conf-micro-fill ${color}-fill" style="width:0%"
                     data-target="${pct}%"></div>
            </div>
            <span class="conf-label">${pct}% confidence</span>
        </div>
    `;

    return card;
}

/**
 * Render (or re-render) the comment list with optional filter.
 * Animates each card in with staggered GSAP tweens.
 * @param {{ text: string, label: string, scores: object }[]} items
 */
function renderCommentList(items) {
    const filtered = activeFilter === 'all'
        ? items
        : items.filter(r => r.label === activeFilter);

    commentList.innerHTML = '';

    if (filtered.length === 0) {
        commentList.innerHTML = '<div class="empty-list">No comments in this category.</div>';
        return;
    }

    // Only show first 30 for performance
    const visible = filtered.slice(0, 30);

    visible.forEach((item, idx) => {
        const card = buildCommentCard(item, idx);
        commentList.appendChild(card);
    });

    // GSAP stagger entrance
    gsap.fromTo(
        commentList.querySelectorAll('.comment-card'),
        { opacity: 0, y: 12 },
        {
            opacity: 1,
            y: 0,
            duration: 0.35,
            stagger: 0.04,
            ease: 'power2.out',
            delay: 0.05,
            onComplete() {
                // Animate each micro confidence bar after cards are visible
                commentList.querySelectorAll('.conf-micro-fill').forEach(el => {
                    const target = el.dataset.target;
                    gsap.to(el, { width: target, duration: 0.7, ease: 'power2.out', delay: 0.1 });
                });
            }
        }
    );
}


/* ─────────────────────────────────────────────
   9. FILTER TABS
   ───────────────────────────────────────────── */

filterTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        // Deactivate all
        filterTabs.forEach(t => t.classList.remove('active'));

        // Activate clicked
        tab.classList.add('active');
        activeFilter = tab.dataset.filter;

        // Animate tab click
        gsap.fromTo(tab,
            { scale: 0.92 },
            { scale: 1, duration: 0.3, ease: 'back.out(2)' }
        );

        renderCommentList(allComments);
    });
});


/* ─────────────────────────────────────────────
   10. BOTTOM NAV SWITCHING
   ───────────────────────────────────────────── */

function switchTab(targetScreen) {
    currentScreen = targetScreen;

    // Update nav active states
    bnavAnalyze.classList.toggle('active', targetScreen === 'analyze');
    bnavManual.classList.toggle('active',  targetScreen === 'manual');

    if (targetScreen === 'manual') {
        showScreen('manual');
        gsap.fromTo(screenManual,
            { opacity: 0, y: 14 },
            { opacity: 1, y: 0, duration: 0.38, ease: 'power2.out' }
        );
    } else {
        // Go back to whatever analyze screen was showing
        if (allComments.length > 0) {
            showScreen('results');
        } else {
            showScreen('not-yt');
        }
    }
}

bnavAnalyze.addEventListener('click', () => switchTab('analyze'));
bnavManual.addEventListener('click',  () => switchTab('manual'));


/* ─────────────────────────────────────────────
   11. MANUAL MODE ANALYSIS
   ───────────────────────────────────────────── */

manualAnalyzeBtn.addEventListener('click', () => {
    const text = manualTextarea.value.trim();
    if (!text) {
        // Shake animation on empty input
        gsap.fromTo(manualTextarea,
            { x: -6 },
            { x: 0, duration: 0.4, ease: 'elastic.out(1,0.3)' }
        );
        return;
    }

    const result = analyzeSentiment(text);
    displayManualResult(result, text);
});

/**
 * Display the sentiment result in Manual Mode UI.
 * @param {{ label: string, scores: object }} result
 * @param {string} text
 */
function displayManualResult(result, text) {
    const color = COLOR_CLASS[result.label];
    const emoji = EMOJI[result.label];
    const conf  = (result.scores[result.label] * 100).toFixed(1);

    // Badge
    manualBadge.innerHTML = `
        <div class="manual-badge-emoji">${emoji}</div>
        <div class="manual-badge-label" style="color:var(--${color === 'green' ? 'green' : color === 'amber' ? 'amber' : 'red'})">${capitalise(result.label)}</div>
        <div class="manual-badge-conf">${conf}% confidence</div>
    `;

    // Confidence bars
    manualConfBars.innerHTML = `
        <div class="manual-bar-row">
            <span>😊 Positive</span>
            <div class="bar-track"><div class="bar-fill green-fill" id="mb-pos" style="width:0%"></div></div>
            <span class="bar-pct-val">${(result.scores.positive * 100).toFixed(0)}%</span>
        </div>
        <div class="manual-bar-row">
            <span>😐 Neutral</span>
            <div class="bar-track"><div class="bar-fill amber-fill" id="mb-neu" style="width:0%"></div></div>
            <span class="bar-pct-val">${(result.scores.neutral * 100).toFixed(0)}%</span>
        </div>
        <div class="manual-bar-row">
            <span>😞 Negative</span>
            <div class="bar-track"><div class="bar-fill red-fill" id="mb-neg" style="width:0%"></div></div>
            <span class="bar-pct-val">${(result.scores.negative * 100).toFixed(0)}%</span>
        </div>
    `;

    manualResult.classList.remove('hidden');

    // Animate in
    gsap.fromTo(manualResult,
        { opacity: 0, scale: 0.97 },
        { opacity: 1, scale: 1, duration: 0.4, ease: 'back.out(1.5)' }
    );

    // Animate bars
    gsap.to('#mb-pos', { width: (result.scores.positive * 100) + '%', duration: 0.9, ease: 'power3.out', delay: 0.2 });
    gsap.to('#mb-neu', { width: (result.scores.neutral  * 100) + '%', duration: 0.9, ease: 'power3.out', delay: 0.3 });
    gsap.to('#mb-neg', { width: (result.scores.negative * 100) + '%', duration: 0.9, ease: 'power3.out', delay: 0.4 });
}


/* ─────────────────────────────────────────────
   12. COMMUNICATION WITH CONTENT SCRIPT
   ───────────────────────────────────────────── */

/**
 * Check if the active tab is a YouTube watch page.
 * If yes, request comments from the content script.
 */
async function init() {
    let tabs;
    try {
        tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    } catch (e) {
        setStatus('inactive', 'Not on YouTube');
        showScreen('not-yt');
        return;
    }

    const tab = tabs[0];
    const isYT = tab && tab.url && tab.url.includes('youtube.com/watch');

    if (!isYT) {
        setStatus('inactive', 'Not on YouTube');
        showScreen('not-yt');
        return;
    }

    setStatus('active', 'YouTube • Live');
    showScreen('loading');

    // Ask content script for comments
    try {
        const response = await chrome.tabs.sendMessage(tab.id, { action: 'GET_COMMENTS' });

        if (response && response.comments && response.comments.length > 0) {
            const analyzed = response.comments.map(text => ({
                text,
                ...analyzeSentiment(text),
            }));

            renderResults(analyzed);
            setStatus('active', `${analyzed.length} comments`);
        } else {
            // Fallback: show demo data if no comments scraped yet (page might still be loading)
            const demoData = getDemoComments();
            const analyzed = demoData.map(text => ({ text, ...analyzeSentiment(text) }));
            renderResults(analyzed);
            setStatus('inactive', 'Demo data (scroll page first)');
        }
    } catch (err) {
        // Content script not reachable
        const demoData = getDemoComments();
        const analyzed = demoData.map(text => ({ text, ...analyzeSentiment(text) }));
        renderResults(analyzed);
        setStatus('inactive', 'Demo mode');
    }
}

/**
 * Set the live status indicator.
 * @param {'active'|'inactive'|'error'} state
 * @param {string} label
 */
function setStatus(state, label) {
    elLiveDot.className   = 'live-dot ' + state;
    elLiveLabel.textContent = label;
}


/* ─────────────────────────────────────────────
   13. DEMO COMMENTS (fallback data)
   ───────────────────────────────────────────── */

function getDemoComments() {
    return [
        "This is absolutely amazing! I've watched it 3 times already.",
        "Not bad, pretty interesting concept overall.",
        "Worst tutorial I've ever seen. Complete waste of time.",
        "Really helpful, thanks so much for explaining it so clearly!",
        "The video is okay, nothing special but does the job.",
        "I love how you break down complex topics so easily.",
        "This is terrible. The audio quality is horrible.",
        "Brilliant explanation! Finally someone who makes sense.",
        "Meh, it's alright. Could be better.",
        "Absolutely incredible work! Subscribed immediately.",
        "This video is so boring and way too long.",
        "Super informative and well structured. Well done!",
        "Not sure what the hype is about, seems very basic.",
        "Great content as always, keep it up!",
        "Disappointing, expected much more from this channel.",
        "Really enjoyed this, it was very insightful.",
        "This is just clickbait. Very misleading title.",
        "Fantastic tutorial, everything worked perfectly for me.",
        "Average content, nothing you can't find elsewhere.",
        "The best explanation I've found on YouTube. Thank you!",
    ];
}


/* ─────────────────────────────────────────────
   14. UTILITY HELPERS
   ───────────────────────────────────────────── */

/**
 * Escape HTML special chars to prevent XSS.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/**
 * Capitalise first letter of a string.
 * @param {string} str
 * @returns {string}
 */
function capitalise(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}


/* ─────────────────────────────────────────────
   15. ENTRY POINT — run when DOM is ready
   ───────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {

    // Animate header in
    gsap.fromTo('.header',
        { opacity: 0, y: -8 },
        { opacity: 1, y: 0, duration: 0.4, ease: 'power2.out' }
    );

    // Animate bottom nav in
    gsap.fromTo('.bottom-nav',
        { opacity: 0, y: 8 },
        { opacity: 1, y: 0, duration: 0.4, ease: 'power2.out', delay: 0.1 }
    );

    // Start the main flow
    init();
});
