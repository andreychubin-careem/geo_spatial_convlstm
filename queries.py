def get_locations_data(date: str, service_area: int = 1) -> str:
    return f"""
    with ios_radar_calls as (
        SELECT
            concat('i', userid) as userid,
            concat(
                '{date} ',
                cast(date_add('millisecond', cast(timestamp as bigint), TIME '00:00:00.000') as varchar)
            ) as ts,
            cast(latitude as double) as latitude,
            cast(longitude as double) as longitude
        FROM app_events.icma
        WHERE date = '{date}'
            and eventname = 'radar_call'
            and service_area_id = '{service_area}'
    ),
    ----------------------------------------------------------------------------------------------------------------------------
    android_radar_calls as (
        SELECT
            concat('a', userid) as userid,
            concat(
                '{date} ',
                cast(date_add('millisecond', cast(timestamp as bigint), TIME '00:00:00.000') as varchar)
            ) as ts,
            cast(latitude as double) as latitude,
            cast(longitude as double) as longitude
        FROM app_events.acma
        WHERE date = '{date}'
            and eventname = 'radar_call'
            and service_area_id = '{service_area}'
    ),
    ----------------------------------------------------------------------------------------------------------------------------
    data as (
        select *
        from ios_radar_calls as a
        union all
        select *
        from android_radar_calls as b
    )
    
    select *
    from data
    where 1=1
        and userid is not null
        and ts is not null
        and latitude is not null
        and longitude is not null
    """
