document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("search-input");
  const grid = document.getElementById("projects-grid");
  if (!input) return;
  fetch(`${document.body.dataset.baseurl || ""}/search.json`)
    .then((r) => r.json())
    .then((data) => {
      const idx = lunr(function () {
        this.ref("url");
        this.field("title");
        this.field("content");
        data.forEach((doc) => this.add(doc));
      });
      input.addEventListener("input", function () {
        const q = input.value.trim();
        if (!grid) return;
        const cards = Array.from(grid.querySelectorAll(".card"));
        if (!q) {
          cards.forEach((c) => (c.style.display = ""));
          return;
        }
        const hits = idx.search(q).map((h) => h.ref);
        cards.forEach((card) => {
          const links = Array.from(card.querySelectorAll("a.nav-link")).map(
            (a) => a.getAttribute("href")
          );
          const match = links.some((u) => hits.includes(u));
          card.style.display = match ? "" : "none";
        });
      });
    });
});
