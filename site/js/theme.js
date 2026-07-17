// theme.js — light/dark перемикач, зберігає вибір локально. Не-модуль (грузиться перший).
(function () {
  var KEY = "skyrun.theme";
  var saved = localStorage.getItem(KEY);
  if (saved) document.documentElement.dataset.theme = saved;
  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("theme");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var cur = document.documentElement.dataset.theme;
      var next = cur === "dark" ? "light" : (cur === "light" ? "dark"
        : (matchMedia("(prefers-color-scheme: dark)").matches ? "light" : "dark"));
      document.documentElement.dataset.theme = next;
      localStorage.setItem(KEY, next);
    });
  });
})();
