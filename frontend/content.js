/* ============================================================
   content.js — YouTube Comment Scraper
   Injected into every youtube.com/watch page.
   Listens for messages from popup.js and replies with comments.
   ============================================================ */

'use strict';

/**
 * Scrape visible comment text nodes from the YouTube DOM.
 * YouTube renders comments lazily inside #comments > ytd-comment-thread-renderer
 * The actual text lives inside yt-formatted-string#content-text
 *
 * @returns {string[]} Array of comment text strings (trimmed, non-empty)
 */
function scrapeComments() {
    const commentNodes = document.querySelectorAll(
        'ytd-comment-thread-renderer #content-text'
    );

    const comments = [];

    commentNodes.forEach(node => {
        // Get the full innerText, collapse whitespace
        const text = node.innerText
            .replace(/\s+/g, ' ')
            .trim();

        // Only include comments with at least 4 characters
        if (text && text.length >= 4) {
            comments.push(text);
        }
    });

    return comments;
}

/* ─── Listen for popup messages ─── */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

    if (message.action === 'GET_COMMENTS') {
        const comments = scrapeComments();

        // Send back to popup
        sendResponse({ comments });
    }

    // Return true so the async sendResponse is valid
    return true;
});
