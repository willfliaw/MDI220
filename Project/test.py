# %%
mean_region = np.mean(df_region["consumption"])
std_region = np.std(df_region["consumption"], ddof=1)

print("Mean:\t\t\t%.5fMW" % mean_region)
print("Standard deviation:\t%.5fMW" % std_region)

# %%
np.random.seed(0)
df_region_generated = norm(loc=mean_region, scale=std_region).rvs(
    size=df_region.shape[0]
)

# %%
plt.figure(figsize=(10, 8), dpi=80)

sns.kdeplot(data=df_region, x="consumption", label="Real data", color="blue", fill=True)
sns.kdeplot(
    data=df_region_generated,
    label="Generated data (Gaussian model)",
    color="red",
    fill=True,
)
plt.title("Kernel Density Estimate (KDE) - Bretagne")
plt.xlabel("Consumption (MW)")
plt.legend()

plt.xticks(fontsize=12, alpha=0.7)
plt.yticks(fontsize=12, alpha=0.7)
plt.grid(axis="both", alpha=0.3)
plt.gca().spines["top"].set_alpha(0.0)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)
plt.gca().spines["left"].set_alpha(0.3)

plt.show()


# %%
def getWinter(df_region):
    """Gets all winter days on a same year (and NOT the days of the winter that started on a year and finished on the next one)"""

    df_region_winter = df_region.copy()
    df_region_winter["date"] = pd.to_datetime(df_region_winter["date"])
    df_region_winter["year"] = df_region_winter["date"].dt.year
    df_region_winter["month"] = df_region_winter["date"].dt.month
    df_region_winter["day"] = df_region_winter["date"].dt.day

    df_region_winter = df_region_winter[
        np.logical_or.reduce(
            (
                np.in1d(df_region_winter["month"], [1, 2]),
                np.logical_and(
                    df_region_winter["month"] == 12, df_region_winter["day"] >= 22
                ),
                np.logical_and(
                    df_region_winter["month"] == 3, df_region_winter["day"] <= 21
                ),
            )
        )
    ]

    return df_region_winter


df_region_winter = getWinter(df_region)

# %%

mean_region_winter = np.mean(df_region_winter["consumption"])
std_region_winter = np.std(df_region_winter["consumption"], ddof=1)

print("Mean:\t\t\t%.5fMW" % mean_region_winter)
print("Standard deviation:\t%.5fMW" % std_region_winter)

# %%

np.random.seed(0)
df_region_winter_generated = norm(loc=mean_region_winter, scale=std_region_winter).rvs(
    size=df_region_winter.shape[0]
)

# %%
plt.figure(figsize=(10, 8), dpi=80)

sns.kdeplot(
    data=df_region_winter, x="consumption", label="Real data", color="blue", fill=True
)
sns.kdeplot(
    data=df_region_winter_generated,
    label="Generated data (Gaussian model)",
    color="red",
    fill=True,
)
plt.title("Kernel Density Estimate (KDE) - Bretagne on Winter")
plt.xlabel("Consumption (MW)")
plt.legend()

plt.xticks(fontsize=12, alpha=0.7)
plt.yticks(fontsize=12, alpha=0.7)
plt.grid(axis="both", alpha=0.3)
plt.gca().spines["top"].set_alpha(0.0)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)
plt.gca().spines["left"].set_alpha(0.3)

plt.show()


# %%
def dissimilarityLebesgue(x, f_x, g_x):
    return 0.5 * np.trapz(np.abs(f_x - g_x), x)


x = np.linspace(
    start=np.min(df_region["consumption"]),
    stop=np.max(df_region["consumption"]),
    num=df_region.shape[0],
)
f_x = gaussian_kde(df_region["consumption"])(x)
g_x = norm(loc=mean_region, scale=std_region).pdf(x)

lebesgue_measure_region = dissimilarityLebesgue(x, f_x, g_x)

print("Dissimilarity (Lebesgue measure): %.5f" % lebesgue_measure_region)

# %%
x = np.linspace(
    start=np.min(df_region_winter["consumption"]),
    stop=np.max(df_region_winter["consumption"]),
    num=df_region_winter.shape[0],
)
f_x = gaussian_kde(df_region_winter["consumption"])(x)
g_x = norm(loc=mean_region_winter, scale=std_region_winter).pdf(x)

lebesgue_measure_region_winter = dissimilarityLebesgue(x, f_x, g_x)

print("Dissimilarity (Lebesgue measure): %.5f" % lebesgue_measure_region_winter)


# %%
def getDissimilarityRegionWinter(region):
    df_region = df[df["region"] == region]

    mean_region = np.mean(df_region["consumption"])
    std_region = np.std(df_region["consumption"], ddof=1)

    x = np.linspace(
        np.min(df_region["consumption"]),
        np.max(df_region["consumption"]),
        num=df_region.shape[0],
    )
    f_x = gaussian_kde(df_region["consumption"])(x)
    g_x = norm(loc=mean_region, scale=std_region).pdf(x)

    lebesgue_measure_region = dissimilarityLebesgue(x, f_x, g_x)

    df_region_winter = getWinter(df_region)

    mean_region_winter = np.mean(df_region_winter["consumption"])
    std_region_winter = np.std(df_region_winter["consumption"], ddof=1)

    x = np.linspace(
        np.min(df_region_winter["consumption"]),
        np.max(df_region_winter["consumption"]),
        num=df_region_winter.shape[0],
    )
    f_x = gaussian_kde(df_region_winter["consumption"])(x)
    g_x = norm(loc=mean_region_winter, scale=std_region_winter).pdf(x)

    lebesgue_measure_region_winter = dissimilarityLebesgue(x, f_x, g_x)

    return lebesgue_measure_region, lebesgue_measure_region_winter


# %%
df_dissimiarities = pd.DataFrame(
    [
        (region, *getDissimilarityRegionWinter(region))
        for region in df["region"].unique()
    ],
    columns=["region", "dissimilarity", "dissimilarity winter"],
)


# %%
df_dissimiarities

# %%
least_dissimilar_region = df_dissimiarities.iloc[
    np.argmin(df_dissimiarities["dissimilarity winter"])
]["region"]

print(
    "The region in which there is a smallest dissimilarity (Lebesgue measure) is %s"
    % least_dissimilar_region
)

# %%
region = df_dissimiarities.iloc[np.argmin(df_dissimiarities["dissimilarity winter"])][
    "region"
]
df_region = df[df["region"] == region]
df_region_winter = getWinter(df_region)

mean_region_winter = np.mean(df_region_winter["consumption"])
std_region_winter = np.std(df_region_winter["consumption"], ddof=1)

np.random.seed(0)
df_region_winter_generated = norm(loc=mean_region_winter, scale=std_region_winter).rvs(
    size=df_region_winter.shape[0]
)

# %%
plt.figure(figsize=(10, 8), dpi=80)

sns.kdeplot(
    data=df_region_winter, x="consumption", label="Real data", color="blue", fill=True
)
sns.kdeplot(
    data=df_region_winter_generated,
    label="Generated data (Gaussian model)",
    color="red",
    fill=True,
)
plt.title("Kernel Density Estimate (KDE) - %s on Winter" % region)
plt.xlabel("Consumption (MW)")
plt.legend()

plt.xticks(fontsize=12, alpha=0.7)
plt.yticks(fontsize=12, alpha=0.7)
plt.grid(axis="both", alpha=0.3)
plt.gca().spines["top"].set_alpha(0.0)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)
plt.gca().spines["left"].set_alpha(0.3)

plt.show()


# %%
def getPosteriorParameters(year, sigma, mu_0, sigma_0):
    region = "Bretagne"
    df_region = df[df["region"] == region]
    df_region_winter = getWinter(df_region)

    mean_region_winter = np.mean(df_region_winter["consumption"])
    std_region_winter = np.std(df_region_winter["consumption"], ddof=1)

    np.random.seed(0)
    df_region_winter_generated = norm(
        loc=mean_region_winter, scale=std_region_winter
    ).rvs(size=df_region_winter.shape[0])

    df_region_winter_year = df_region_winter[df_region_winter["year"] == year]

    std2_n = (
        (df_region_winter_year.shape[0] / (sigma**2)) + (1 / (sigma_0**2))
    ) ** (-1)
    mu_n = std2_n * (
        (np.sum(df_region_winter_year["consumption"]) / (sigma**2))
        + (mu_0 / (sigma_0**2))
    )

    return mu_n, std2_n, np.sqrt(std2_n)


# %%
sigma = 400
mu_0 = 3500
sigma_0 = 500

df_posteriorParameters = pd.DataFrame(
    [
        (year, *getPosteriorParameters(year, sigma, mu_0, sigma_0))
        for year in range(2013, 2024)
    ],
    columns=["year", "mean", "variance", "standard deviation"],
)

# %%
df_posteriorParameters

# %%
year = 2023
region = "Bretagne"

df_region = df[df["region"] == region]
df_region_winter = getWinter(df_region)
df_region_winter_year = df_region_winter[df_region_winter["year"] == year]

std2_n = ((df_region_winter_year.shape[0] / (sigma**2)) + (1 / (sigma_0**2))) ** (
    -1
)
mu_n = std2_n * (
    (np.sum(df_region_winter_year["consumption"]) / (sigma**2))
    + (mu_0 / (sigma_0**2))
)

print("mu_n:\t%.5fMW" % mu_n)
print("std2_n:\t%.5fMW" % std2_n)

np.random.seed(0)
df_region_winter_year_posterior_generated = norm(loc=mu_n, scale=np.sqrt(std2_n)).rvs(
    size=df_region_winter_year.shape[0]
)

# %%
plt.figure(figsize=(10, 8), dpi=80)

sns.kdeplot(
    data=df_region_winter_year,
    x="consumption",
    label="Real data",
    color="blue",
    fill=True,
)
sns.kdeplot(
    data=df_region_winter_year_posterior_generated,
    label="Generated data (Posterior distribution)",
    color="red",
    fill=True,
)
plt.title("Kernel Density Estimate (KDE) - %s on Winter %d" % (region, year))
plt.xlabel("Consumption (MW)")
plt.legend()

plt.xticks(fontsize=12, alpha=0.7)
plt.yticks(fontsize=12, alpha=0.7)
plt.grid(axis="both", alpha=0.3)
plt.gca().spines["top"].set_alpha(0.0)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)
plt.gca().spines["left"].set_alpha(0.3)

plt.show()

# %%
x = np.linspace(
    start=np.min(df_region_winter_year["consumption"]),
    stop=np.max(df_region_winter_year["consumption"]),
    num=df_region_winter_year.shape[0],
)
f_x = gaussian_kde(df_region_winter_year["consumption"])(x)
g_x = norm(loc=mu_n, scale=np.sqrt(std2_n)).pdf(x)

lebesgue_measure_region_winter_year_posterior = dissimilarityLebesgue(x, f_x, g_x)

print(
    "Dissimilarity (Lebesgue measure): %.5f"
    % lebesgue_measure_region_winter_year_posterior
)

# %%
year = 2023
region = "Bretagne"

df_region = df[df["region"] == region]
df_region_winter = getWinter(df_region)
df_region_winter_year = df_region_winter[df_region_winter["year"] == year]

alpha = 0.01
sigma_0 = 400
sigma_1 = 500
mu = 3200

S = np.sum(np.power(df_region["consumption"] - mu, 2))

# %%
"""Two-tailed hypothesis"""

c_1 = (sigma_0**2) * chi2(df=df_region_winter_year.shape[0]).ppf(q=alpha / 2)
c_2 = (sigma_0**2) * chi2(df=df_region_winter_year.shape[0]).ppf(q=1 - (alpha / 2))

print("S:\t%.5f " % S)
print("c_1:\t%.5f " % c_1)
print("c_2:\t%.5f " % c_2)
print(
    "As S is not in the interval [c1, c2], we reject the null hypothesis: the standard deviation of the consumption in 2023 is not 400MW."
    if S < c_1 or S > c_2
    else "As S is in the interval [c1, c2], we accept the null hypothesis: the standard deviation of the consumption in 2023 is 400MW."
)

# %%
"""Simple Hypothesis"""

c = (sigma_0**2) * chi2(df=df_region_winter_year.shape[0]).ppf(q=1 - alpha)

print("S:\t%.5f " % S)
print("c:\t%.5f " % c)
print(
    "As S is greater than c, we reject the null hypothesis: the standard deviation of the consumption in 2023 is not 400MW."
    if S > c
    else "As S is smaller than c, we accept the null hypothesis: the standard deviation of the consumption in 2023 is 400MW."
)

# %%
region1 = "Bretagne"
df_bretagne = df[df["region"] == region1]
df_bretagne_winter = getWinter(df_bretagne)

region2 = "Provence-Alpes-Côte d'Azur"
df_paca = df[df["region"] == region2]
df_paca_winter = getWinter(df_paca)

# %%
plt.figure(figsize=(8, 8), dpi=80)

sns.jointplot(
    x=list(df_bretagne_winter["consumption"]),
    y=list(df_paca_winter["consumption"]),
    kind="reg",
    height=8,
)

plt.xlabel("%s" % region1)
plt.ylabel("%s" % region2)

plt.xticks(fontsize=12, alpha=0.7)
plt.yticks(fontsize=12, alpha=0.7)
plt.grid(axis="both", alpha=0.3)

plt.show()

# %%
alpha = 0.01
df_region12_winter = pd.DataFrame(
    [
        df_bretagne_winter["consumption"].to_numpy(),
        df_paca_winter["consumption"].to_numpy(),
    ],
    index=[region1, region2],
).T

bins = 10
H, xedges, yedges = np.histogram2d(
    x=df_region12_winter[region1], y=df_region12_winter[region2], bins=bins
)

Nis = np.sum(H, axis=0)
Njs = np.sum(H, axis=1)
n = np.sum(H)

T = np.nansum(
    [
        ((H[i, j] - (Nis[i] * Njs[j] / n)) ** 2) / (Nis[i] * Njs[j] / n)
        for j in range(bins)
        for i in range(bins)
    ]
)
c = chi2(df=(bins - 1) ** 2).ppf(q=1 - alpha)

print("T:\t%.5f" % T)
print("c:\t%.5f" % c)
print(
    "As T is greater than c, we reject the null hypothesis: the consumption in the two regions is not independent."
    if T > c
    else "As T is smaller than c, we accept the null hypothesis: the consumption in the two regions is independent."
)

# %%
region = "Bretagne"
df_region = df[df["region"] == region]
df_region_winter = getWinter(df_region)

n = df_region_winter["consumption"].shape[0]
mean = np.mean(df_region_winter["consumption"])
std = np.std(df_region_winter["consumption"], ddof=1)

print("Mean:\t\t\t%.5fMW" % mean)
print("Standard deviation:\t%.5fMW" % std)

alpha = 0.95

# %%
"""Symmetric confidence interval"""

c = norm(loc=0, scale=1).ppf((1 - alpha) / 2)
delta = c * std / np.sqrt(n)

print("Confidence interval:\t [%.5f, %.5f] MW" % (mean - delta, mean + delta))

# %%
df_region_winter_year = df_region_winter[df_region_winter["year"] == 2023]

n = df_region_winter_year["consumption"].shape[0]
mean = np.mean(df_region_winter_year["consumption"])
std = np.std(df_region_winter_year["consumption"], ddof=1)

print("Mean:\t\t\t%.5fMW" % mean)
print("Standard deviation:\t%.5fMW" % std)

alpha = 0.05
mu = 3100

# %%
"""Symmetric confidence interval"""

c = norm(loc=0, scale=1).ppf(1 - alpha / 2)
delta = c * std / np.sqrt(n)

print("c:\t%.5f" % c)
print("delta:\t%.5fMW" % delta)
print(
    "As the mean is not in the interval [%.5f, %.5f], we reject the null hypothesis: the mean consumption in 2023 is not equal to 3100MW."
    % (mu - delta, mu + delta)
    if mean < mu - delta or mean > mu + delta
    else "As the mean is in the interval [%.5f, %.5f], we accept the null hypothesis: the mean consumption in 2023 is equal to 3100MW."
    % (mu - delta, mu + delta)
)
