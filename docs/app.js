const menuButton = document.querySelector('.menu-button');
const nav = document.querySelector('nav');

menuButton?.addEventListener('click', () => {
  const open = nav.classList.toggle('is-open');
  menuButton.setAttribute('aria-expanded', String(open));
});

nav?.addEventListener('click', (event) => {
  if (event.target.closest('a')) {
    nav.classList.remove('is-open');
    menuButton?.setAttribute('aria-expanded', 'false');
  }
});
