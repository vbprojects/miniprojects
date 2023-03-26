trial() = begin
    x = rand()
    for _ in 1:5000
        x += rand() - rand()
    end
    x
end

[trial() for _ in 1:10000] |> mean