/* ============================================================
   background.js — YouTube Comment Scraper Extension
   Service worker for background tasks if needed later.
   ============================================================ */

'use strict';

chrome.runtime.onInstalled.addListener(() => {
    console.log('YT Sentiment Analyzer extension installed.');
});
