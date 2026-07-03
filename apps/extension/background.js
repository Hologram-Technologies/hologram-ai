chrome.runtime.onInstalled.addListener(() => {
  console.log("Holospaces Egress Extension installed.");
  updateAuthRule();
});

chrome.cookies.onChanged.addListener((changeInfo) => {
  if (changeInfo.cookie.domain.includes("huggingface.co") && changeInfo.cookie.name === "token") {
    updateAuthRule();
  }
});

function updateAuthRule() {
  chrome.cookies.get({ url: "https://huggingface.co", name: "token" }, (cookie) => {
    if (cookie) {
      const token = cookie.value;
      chrome.declarativeNetRequest.updateDynamicRules({
        removeRuleIds: [1],
        addRules: [{
          id: 1,
          priority: 1,
          action: {
            type: "modifyHeaders",
            requestHeaders: [
              { header: "Authorization", operation: "set", value: "Bearer " + token }
            ]
          },
          condition: {
            urlFilter: "||huggingface.co",
            excludedRequestDomains: ["cdn-lfs.huggingface.co", "cdn-lfs-us-1.huggingface.co", "cdn-lfs-us-2.huggingface.co"],
            resourceTypes: ["xmlhttprequest"]
          }
        }]
      });
      console.log("Auth rule updated with token.");
    } else {
      chrome.declarativeNetRequest.updateDynamicRules({
        removeRuleIds: [1]
      });
      console.log("Auth rule removed (no token found).");
    }
  });
}

// Initial update
updateAuthRule();
