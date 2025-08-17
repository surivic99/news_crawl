import time
import logging
import json
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Source Data (French) ---
french_queries = {
    "compte étudiant bgl": ["https://www.bgl.lu/fr/particuliers/comptes-bancaires/compte-bancaire-jeune.html", "https://bgl.lu/fr/particuliers/comptes-bancaires/compte-bancaire-jeunes.html", "https://www.bgl.lu/fr/particuliers/epargner/compte-epargne-jeunes.html", "https://www.bgl.lu/fr/particuliers/projet-etudes/pret-etudiant-au-luxembourg.html"],
    "plafonds carte bgl": ["https://www.bgl.lu/fr/particuliers/cartes-de-paiement/difference-carte-credit-debit.html", "http://bgl.lu/fr/particuliers/cartes-de-paiement/visa-classic.html", "https://www.bgl.lu/fr/particuliers/cartes-de-paiement.html", "https://www.bgl.lu/content/dam/publicsite/pdf/documents-officiels/conditions-cartes/v-pay/Conditions-d-utilisation-V-PAY-FR.pdf", "https://bgl.lu/fr/particuliers/cartes-de-paiement/visa-debit.html", "https://www.bgl.lu/fr/particuliers/cartes-de-paiement/mastercard-platinum.html"],
    "pack bancaire bgl": ["https://www.bgl.lu/fr/particuliers/comptes-bancaires/compte-bancaire-essentiel.html", "https://www.bgl.lu/fr/particuliers/comptes-bancaires/compte-bancaire-exclusif.html", "https://www.bgl.lu/fr/particuliers/comptes-bancaires/compte-bancaire-jeune.html"],
    "tarifs cartes bgl": ["https://www.bgl.lu/content/dam/publicsite/pdf/documents-officiels/tarif/guide-des-tarifs/guide-tarifaire.pdf", "https://www.bgl.lu/content/dam/publicsite/pdf/brochures/particuliers/moyens-de-paiement/bien-choisir-sa-carte/bien-choisir-sa-carte.pdf", "https://www.bgl.lu/content/dam/publicsite/pdf/documents-officiels/tarif/guide-des-tarifs/guide-des-tarifs-Pro-fr.pdf", "https://www.bgl.lu/fr/documents-officiels/brochures-et-tarifs.html"],
    "prêt à taux variable/fixe bgl": ["https://www.bgl.lu/fr/particuliers/blog/immobilier/taux-fixe-variable-revisable.html", "https://bgl.lu/fr/particuliers/projet-immo/pret-immobilier.html", "https://bgl.lu/fr/particuliers/projet-immo/simulation.html", "https://www.bgl.lu/content/dam/publicsite/pdf/brochures/particuliers/financer-votre-projet/pr%C3%AAt-immo/conditions-immo/Conditions-immo-FR.pdf"],
    "ouvrir compte bgl non-résident": ["https://www.bgl.lu/fr/particuliers/comptes-bancaires/comment-ouvrir-compte-bancaire.html", "https://www.bgl.lu/fr/particuliers/comptes-bancaires/demande-ouverture-compte.html", "https://www.bgl.lu/fr/particuliers/comptes-bancaires.html", "https://www.bgl.lu/fr/particuliers/comptes-bancaires/compte-en-agence.html"],
    "compte pro bgl": ["https://www.bgl.lu/fr/entreprises/compte-pro.html", "https://bgl.lu/fr/entreprises/independants.html", "https://www.bgl.lu/fr/entreprises.html", "https://bgl.lu/fr/entreprises/professions-liberales.html"],
    "bgl propose-t-elle l’apple pay": ["https://www.bgl.lu/fr/particuliers/paiements-mobiles/apple-pay.html", "https://www.bgl.lu/fr/particuliers/services-en-ligne/web-banking/paiements.html"],
    "incident de paiement bgl": ["https://www.bgl.lu/fr/particuliers/services-en-ligne/web-banking/paiements.html", "https://www.bgl.lu/fr/particuliers/securite-en-ligne/alerte-fraude.html", "https://www.bgl.lu/fr/webbanking", "https://www.bgl.lu/fr/particuliers/services-en-ligne/web-banking/comptes.html"],
    "compte mineur bgl": ["https://www.bgl.lu/fr/particuliers/comptes-bancaires/comment-ouvrir-compte-bancaire.html", "https://www.bgl.lu/fr/particuliers/epargner/compte-epargne-jeunes.html", "https://www.bgl.lu/fr/particuliers/epargner/compte-epargne-croissance.html", "https://www.bgl.lu/fr/particuliers/comptes-bancaires/compte-bancaire-jeune.html"],
    "bgl produits deductibles des impots": ["https://www.bgl.lu/fr/particuliers/fiscalement-deductible.html", "https://www.bgl.lu/fr/particuliers/fiscalement-deductible/guide-deductions-fiscales-luxembourg.html", "https://www.bgl.lu/fr/particuliers/fiscalement-deductible/produits-fiscalement-deductibles.html"],
    "profil investisseur bgl": ["https://www.bgl.lu/fr/particuliers/services-en-ligne/web-banking/gerer-mes-investissements.html", "http://bgl.lu/fr/particuliers/investir/conseil-investissement-multi-assets.html", "https://www.bgl.lu/fr/particuliers/investir/selection-de-fonds.html", "https://www.bgl.lu/fr/particuliers/investir/investir-en-ligne.html", "https://www.bgl.lu/fr/particuliers/investir/investir-en-ligne.html", "https://www.bgl.lu/fr/particuliers/epargner/faire-fructifier-epargne.html", "https://bgl.lu/fr/particuliers/investir/epargne-programmee-en-ligne.html", "https://bgl.lu/fr/particuliers/investir.html"],
    "prêt travaux bgl": ["https://www.bgl.lu/fr/particuliers/autres-projets/pret-travaux-au-luxembourg.html", "https://www.bgl.lu/fr/particuliers/projet-renovation/pret-renovation-energetique.html", "https://www.bgl.lu/fr/particuliers/projet-renovation.html", "https://www.bgl.lu/fr/particuliers/autres-projets/pret-personnel.html", "https://www.bgl.lu/fr/particuliers/projet-renovation/aides-renovation-energetique.html", "https://www.bgl.lu/fr/particuliers/projet-renovation/accompagnement-renovation.html"],
    "crédit auto bgl": ["https://www.bgl.lu/fr/particuliers/projet-auto/credit-auto.html", "https://www.bgl.lu/fr/particuliers/projet-auto.html", "https://www.bgl.lu/fr/particuliers/projet-auto/private-lease-credit-auto.html", "https://www.bgl.lu/fr/particuliers/projet-auto/acheter-une-voiture.html", "https://www.bgl.lu/fr/particuliers/projet-auto/credit-voiture-electrique.html"],
    "financement résidence principale bgl": ["https://www.bgl.lu/fr/particuliers/projet-immo/premier-achat-immobilier.html", "https://www.bgl.lu/fr/particuliers/projet-immo/pret-immobilier.html", "https://www.bgl.lu/fr/particuliers/projet-immo/acheter-vendre-logement.html", "https://www.bgl.lu/content/dam/publicsite/pdf/brochures/particuliers/financer-votre-projet/pr%C3%AAt-immo/guide-immo/brochure-guide-immo/Brochure-Guide-Pratique-Immobilier-FR.pdf", "https://www.bgl.lu/fr/particuliers/blog/immobilier/aide-etat-achat-immobilier.html", "https://bgl.lu/fr/particuliers/projet-immo/simulation.html"],
    "activer luxtrust bgl": ["https://bgl.lu/content/dam/publicsite/pdf/brochures/particuliers/banque-au-quotidien/services-en-ligne/FR-guide-de-premiere-connexion.pdf", "https://www.bgl.lu/fr/particuliers/services-en-ligne/luxtrust-mobile.html", "https://www.bgl.lu/fr/particuliers/services-en-ligne/web-banking/connexion.html"],
    "virement instantané bgl": ["https://www.bgl.lu/fr/particuliers/services-en-ligne/virement-instantane.html"],
    "tarifs virements bgl ": ["https://www.bgl.lu/content/dam/publicsite/pdf/documents-officiels/tarif/guide-des-tarifs/guide-tarifaire.pdf", "https://www.bgl.lu/fr/documents-officiels/brochures-et-tarifs.html", "https://www.bgl.lu/fr/documents-officiels/brochures-et-tarifs/changement-de-tarifs.html", "https://www.bgl.lu/content/dam/publicsite/pdf/documents-officiels/fid/compte-courant-hors-essentiel/FID-Compte-courant-hors-Essentiel-FR.pdf"],
    "solutions d'epargne bgl": ["https://www.bgl.lu/fr/particuliers/epargner.html", "https://www.bgl.lu/fr/particuliers/epargner.html", "https://www.bgl.lu/fr/particuliers/epargner/conseils-epargne.html", "https://www.bgl.lu/fr/particuliers/epargner/compte-epargne.html", "https://www.bgl.lu/fr/particuliers/epargner/faire-fructifier-epargne.html", "https://www.bgl.lu/fr/particuliers/epargner/epargne-programmee.html", "https://www.bgl.lu/fr/particuliers/projet-immo/epargne-logement.html"],
    "apple pay bgl": ["https://www.bgl.lu/fr/particuliers/paiements-mobiles/apple-pay.html", "https://www.bgl.lu/fr/particuliers/paiements-mobiles.html"],
    "prêt vélo taux 0 bgl": ["https://bgl.lu/fr/particuliers/comptes-bancaires/compte-bancaire-en-ligne.html"],
    "carte platinum bgl avantages": ["https://www.bgl.lu/fr/particuliers/cartes-de-paiement/mastercard-platinum.html", "https://www.bgl.lu/fr/particuliers/cartes-de-paiement/assurance-ski-mastercard.html", "https://bgl.lu/fr/particuliers/cartes-de-paiement/assurance-voyage-mastercard.html"],
    "optipension bgl": ["https://www.bgl.lu/fr/particuliers/assurances-vie/assurance-retraite-luxembourg.html"],
    "direct invest clic bgl": ["https://bgl.lu/fr/particuliers/investir/epargne-programmee-en-ligne.html"],
    "optisave bgl": ["https://www.bgl.lu/fr/particuliers/assurances-vie/assurance-vie-epargne.html"],
    "contacter service client bgl": ["https://www.bgl.lu/fr/particuliers/contact.html"],
    "web banking bgl fonctionnalités": ["https://www.bgl.lu/fr/particuliers/services-en-ligne/web-banking/connexion.html", "https://www.bgl.lu/fr/particuliers/services-en-ligne/web-banking/comptes.html", "https://www.bgl.lu/fr/particuliers/comptes-bancaires/welcome-client.html"]
}

# --- Pre-translated queries for reliability and speed ---
query_translations = {
    "compte étudiant bgl": {"en": "bgl student account", "de": "BGL Studenten-Konto"},
    "plafonds carte bgl": {"en": "bgl card limits", "de": "BGL Kartenlimits"},
    "pack bancaire bgl": {"en": "bgl banking package", "de": "BGL Bankpaket"},
    "tarifs cartes bgl": {"en": "bgl card fees", "de": "BGL Kartengebühren"},
    "prêt à taux variable/fixe bgl": {"en": "bgl variable/fixed rate loan", "de": "BGL Darlehen mit variablem/festem Zinssatz"},
    "ouvrir compte bgl non-résident": {"en": "open bgl account non-resident", "de": "BGL Konto für Nichtansässige eröffnen"},
    "compte pro bgl": {"en": "bgl professional account", "de": "BGL Geschäftskonto"},
    "bgl propose-t-elle l’apple pay": {"en": "does bgl offer apple pay", "de": "Bietet BGL Apple Pay an"},
    "incident de paiement bgl": {"en": "bgl payment incident", "de": "BGL Zahlungsstörung"},
    "compte mineur bgl": {"en": "bgl account for minors", "de": "BGL Konto für Minderjährige"},
    "bgl produits deductibles des impots": {"en": "bgl tax deductible products", "de": "BGL steuerlich absetzbare Produkte"},
    "profil investisseur bgl": {"en": "bgl investor profile", "de": "BGL Anlegerprofil"},
    "prêt travaux bgl": {"en": "bgl home improvement loan", "de": "BGL Renovierungskredit"},
    "crédit auto bgl": {"en": "bgl car loan", "de": "BGL Autokredit"},
    "financement résidence principale bgl": {"en": "bgl primary residence financing", "de": "BGL Finanzierung des Hauptwohnsitzes"},
    "activer luxtrust bgl": {"en": "activate luxtrust bgl", "de": "Luxtrust BGL aktivieren"},
    "virement instantané bgl": {"en": "bgl instant transfer", "de": "BGL Echtzeit-Überweisung"},
    "tarifs virements bgl ": {"en": "bgl transfer fees", "de": "BGL Überweisungsgebühren"},
    "solutions d'epargne bgl": {"en": "bgl savings solutions", "de": "BGL Sparlösungen"},
    "apple pay bgl": {"en": "apple pay bgl", "de": "Apple Pay BGL"},
    "prêt vélo taux 0 bgl": {"en": "bgl 0% interest bike loan", "de": "BGL 0%-Zins-Fahrradkredit"},
    "carte platinum bgl avantages": {"en": "bgl platinum card benefits", "de": "Vorteile der BGL Platinum-Karte"},
    "optipension bgl": {"en": "optipension bgl", "de": "OptiPension BGL"},
    "direct invest clic bgl": {"en": "direct invest clic bgl", "de": "Direct Invest Clic BGL"},
    "optisave bgl": {"en": "optisave bgl", "de": "OptiSave BGL"},
    "contacter service client bgl": {"en": "contact bgl customer service", "de": "BGL Kundenservice kontaktieren"},
    "web banking bgl fonctionnalités": {"en": "bgl web banking features", "de": "BGL Web-Banking-Funktionen"}
}


def get_language_links(url, driver):
    """
    Scrapes a single URL to find its language variants using the correct selectors.
    Reuses the same driver instance for efficiency.
    """
    try:
        logger.info(f"Scraping for language links on: {url}")
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(2)  # Allow JS to render elements if needed

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        lang_div = soup.find("div", class_="options language-options")

        result_links = {'English': None, 'Deutsch': None, 'Français': None}
        
        if lang_div:
            # *** THIS IS THE CRITICAL FIX ***
            # Using the correct, specific selector from your working script
            for li in lang_div.select("ul.nav-language-wrapper li.nav-element"):
                lang_text = li.get_text(strip=True).upper()
                a_tag = li.find("a", href=True)

                lang_key = None
                if lang_text == 'EN': lang_key = 'English'
                elif lang_text == 'DE': lang_key = 'Deutsch'
                elif lang_text == 'FR': lang_key = 'Français'
                
                if not lang_key: continue
                
                if "active" in li.get("class", []):
                    result_links[lang_key] = url  # The current page is the active one
                elif a_tag:
                    result_links[lang_key] = urljoin(url, a_tag['href'].strip())
        else:
            logger.warning(f"Could not find language switcher div on {url}")
        
        return result_links

    except Exception as e:
        logger.error(f"An error occurred while scraping {url}: {e}")
        return {'English': None, 'Deutsch': None, 'Français': None}


def main():
    driver = None
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        service = EdgeService(executable_path="msedgedriver.exe")
        driver = webdriver.Edge(service=service, options=options)        
        
        all_results = {"fr": french_queries, "en": {}, "de": {}}
        url_cache = {}

        for fr_query, fr_urls in french_queries.items():
            logger.info(f"--- Processing query: '{fr_query}' ---")
            
            translations = query_translations.get(fr_query.strip())
            if not translations:
                logger.warning(f"No translations found for '{fr_query}'. Skipping.")
                continue
            
            en_query, de_query = translations['en'], translations['de']
            en_urls_found, de_urls_found = [], []

            for url in set(fr_urls):
                if not url or not url.startswith('http'):
                    logger.warning(f"Skipping invalid URL: {url}")
                    continue

                if url.endswith('.pdf'):
                    logger.info(f"Handling PDF URL: {url}")
                    en_urls_found.append(url.replace('/fr/', '/en/').replace('-FR.', '-EN.'))
                    de_urls_found.append(url.replace('/fr/', '/de/').replace('-FR.', '-DE.'))
                    continue

                lang_links = url_cache.get(url)
                if not lang_links:
                    lang_links = get_language_links(url, driver)
                    url_cache[url] = lang_links

                if lang_links:
                    if lang_links.get('English'): 
                        en_urls_found.append(lang_links['English'])
                        logger.info(f"Found English link: {lang_links['English']}")
                    if lang_links.get('Deutsch'): 
                        de_urls_found.append(lang_links['Deutsch'])
                        logger.info(f"Found German link: {lang_links['Deutsch']}")

            all_results["en"][en_query] = sorted(list(set(en_urls_found)))
            all_results["de"][de_query] = sorted(list(set(de_urls_found)))

        output_filename = "bgl_queries_multilang.json"
        logger.info(f"Saving all results to {output_filename}")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        
        logger.info("Processing complete!")

    finally:
        if driver:
            driver.quit()
            logger.info("WebDriver has been closed.")

if __name__ == "__main__":
    main()